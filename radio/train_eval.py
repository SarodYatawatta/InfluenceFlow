#! /usr/bin/env python
import argparse
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.functional
import torch.optim as optim
import numpy as np
import generate_data
from training_buffer import TrainingBuffer
from dnn_models import BaseDist
import shapelet_model

# (try to) use a GPU for computation?
use_cuda=True
if use_cuda and torch.cuda.is_available():
  mydevice=torch.device('cuda')
else:
  mydevice=torch.device('cpu')

# epsilon
EPS=1e-6

# problem size=directions ~6 at most gives invertible mapping
K=6
N=14 # stations, fixed
B=N*(N-1)//2 # baselines, fixed
# each epoch, train and evaluate
n_epochs=1
# training iterations
n_iter=250000
# evaluation iterations
n_eval=3000
batch_size=256
n_buffer=15000#0
# input dimension: 8xbaselines, fixed
n_input=B*8
load_buffer=1 # if 1, load, if 0, simulate and save

admm_rho=0.001

constraint_type = 0
residual_error_bound = 0.8

dual_update_cadence=100 # how many iterations for updating Lagrange multiplier 
permute_threshold=1.1 # if a uniform [0,1] rand value is greater than this, permute (uvw,unscaled_uvw,data,residual), larger means less permutation
do_batchnorm=0 # if 1, enable batchnorm 

# scale factors (only applied from simulation to buffer)
# the buffer returns normalized values afterwards
uvw_scale=1/1000
data_scale=100
solution_scale=1
sky_scale=1
residual_scale=100

# loss normalization factors (tune this with admm_rho)
loss1_scale=-1 # maximize loss

# hidden layer dim ~ 8*B
n_hidden=8*B
# depth of layers (total depth x2 + 1 this value)
n_depth=1

# learning rate: start with 0 -> target_lr, thereafter
# switch to cosine annealing till the end
target_lr=1e-4
# where to end warmup, look at wild swings in cost and decide
warmup_steps=5000

load_model=0
save_model=1
do_normalize=False
dropout=False

# if 1, first train the model to predict the residual
initial_training=0

parser = argparse.ArgumentParser(description="Train normalizing flow",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def init_parser():
    parser.add_argument('--K', default=K, type=int, help='Number of directions')
    parser.add_argument('--iterations', default=n_iter, type=int, help='Training iterations')
    parser.add_argument('--warmup', default=warmup_steps, type=int, help='Warmup iterations')
    parser.add_argument('--evaluations', default=n_eval, type=int, help='Evaluation iterations')
    parser.add_argument('--batch_size', default=batch_size, type=int, help='Batch size')
    parser.add_argument('--data_size', default=n_buffer, type=int, help='Training data size')
    parser.add_argument('--learning_rate', default=target_lr, type=float, help='Learning rate')
    parser.add_argument('--load_data', default=load_buffer, type=int, help='Load saved data buffer (if False, data will be generated)')
    parser.add_argument('--load_model', default=load_model, type=int, help='Load saved model (and optimizer state)')
    parser.add_argument('--save_model', default=save_model, type=int, help='Save model (and optimizer state)')
    parser.add_argument('--regularization', default=admm_rho, type=float, help='Regularization factor for constraint')
    parser.add_argument('--constraint_type', default=constraint_type, type=int, help='If 0, constraint x^T y, if 1, constraint ||x-y||^2')
    parser.add_argument('--error_bound', default=residual_error_bound, type=float, help='x^T y > error_bound, set high ~ 1, or ||x - y||^2 < error_bound, set low ~ 0')
    parser.add_argument('--DNN_hidden', default=n_hidden, type=int, help='DNN hidden layer dimension')
    parser.add_argument('--DNN_depth', default=n_depth, type=int, help='DNN layer depth')
    parser.add_argument('--batchnorm', default=do_batchnorm, type=int, help='If 1, enable batchnorm')
    parser.add_argument('--initial_training', default=initial_training, type=int, help='If 1, train the model to predict the residual')


init_parser()
args = parser.parse_args()

# get back command line args
K=args.K
n_iter=args.iterations
warmup_steps=args.warmup
n_eval=args.evaluations
batch_size=args.batch_size
n_buffer=args.data_size
target_lr=args.learning_rate
load_buffer=args.load_data
admm_rho=args.regularization
residual_error_bound=args.error_bound
save_model=args.save_model
load_model=args.load_model
n_hidden=args.DNN_hidden
n_depth=args.DNN_depth
do_batchnorm=args.batchnorm
initial_training=args.initial_training
constraint_type = args.constraint_type

# constraint_type 0 :
# (delta - y^T x / ||x|| ||y||) < 0
# constraint_type 1 :
#  ||x-y||^2 / ||y||^2 - delta < 0
if constraint_type == 0:
   # constraint_type 0, higher is better
   # delta=max value is 1, y^T x/||x||||y||, per each batch
   recon_upper_bound=2 # upper bound of (-y^T x/||x|||y||+delta), per each batch
else:
   # constraint_type 1, lower is better
   recon_upper_bound=2 # upper bound of (||x-y||^2 /||y||^2 - delta), per each batch

# for validation, return log probabilities
def evaluate_model(model,test_buffer):
   n_m_eval=min(batch_size,n_eval)
   sum_log_px=0
   dot_prod=0
   model.eval()
   with torch.no_grad():
       uvw,unscaled_uvw,data,solution,sky,log_eig,residual=test_buffer.sample_buffer(n_m_eval,normalize=do_normalize)
       # no need to clamp all values
       uvw=torch.tensor(uvw).to(mydevice)
       data=torch.tensor(data).to(mydevice)
       solution=torch.tensor(solution).to(mydevice)
       sky=torch.tensor(sky).to(mydevice)
       y=torch.tensor(residual).to(mydevice)
       log_eig=torch.tensor(log_eig).to(mydevice)
       xtilde, log_px =model.sample(y,uvw,data,solution,sky,reparameterize=False)
       sum_log_px +=torch.sum(log_px).detach().cpu().numpy().squeeze()
       dot_prod += torch.sum(torch.bmm(y.unsqueeze(1),xtilde.unsqueeze(2)).squeeze()/(torch.mean(torch.linalg.norm(y,dim=1))*torch.mean(torch.linalg.norm(xtilde,dim=1))+EPS))

   model.train()
   return sum_log_px/(n_m_eval*n_input),dot_prod/n_m_eval

for epoch in range(n_epochs):
   bd=BaseDist(N,K,n_hidden=n_hidden,depth=n_depth,epsilon=1e-4,batchnorm=do_batchnorm,dropout=dropout)
   if load_model:
       bd.load_checkpoint()
   bd=torch.compile(bd)

   total_parameters=bd.parameters()
   buffer=TrainingBuffer(n_buffer,N,K)
   test_buffer=TrainingBuffer(n_eval,N,K)
   # fill buffer
   if not load_buffer:
      for cj in range(n_buffer):
         n_stat, uvw, data, sol, sky, log_eig, residual, _=\
              generate_data.generate_training_data(K)
         # keep original uvw for basis calculation
         unscaled_uvw=uvw.copy()
         uvw *=uvw_scale
         data *=data_scale
         sol *=solution_scale
         sky *=sky_scale
         residual *=residual_scale
         if generate_data.quality_ok(data,residual):
             buffer.store_observation(uvw,unscaled_uvw,data,sol,sky,log_eig,residual)
      buffer.save_checkpoint('buffer_'+str(epoch)+'.npy')
      for cj in range(n_eval):
         n_stat, uvw, data, sol, sky, log_eig, residual, _=\
              generate_data.generate_training_data(K)
         # keep original uvw for basis calculation
         unscaled_uvw=uvw.copy()
         uvw *=uvw_scale
         data *=data_scale
         sol *=solution_scale
         sky *=sky_scale
         residual *=residual_scale
         if generate_data.quality_ok(data,residual):
             test_buffer.store_observation(uvw,unscaled_uvw,data,sol,sky,log_eig,residual)
      test_buffer.save_checkpoint('test_buffer_'+str(epoch)+'.npy')
   else:
      buffer.load_checkpoint('buffer_'+str(epoch)+'.npy')
      test_buffer.load_checkpoint('test_buffer_'+str(epoch)+'.npy')

   #optimizer=optim.Adam(total_parameters,lr=target_lr)
   # AdamW is more stable for long runs
   optimizer=optim.AdamW(total_parameters,lr=target_lr,weight_decay=1e-5)
   #if load_model:
   #    optimizer.load_state_dict(torch.load('opt.state'))

   warmup_lambda=lambda step: min(1.0,(step+1)/warmup_steps)
   warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
   # adding annlealing degrades performance a bit, could improve by increasing n_iter
   decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iter-warmup_steps)
   scheduler=torch.optim.lr_scheduler.SequentialLR(
           optimizer,
           schedulers=[warmup_scheduler, decay_scheduler],
           milestones=[warmup_steps]
           )
   if not initial_training:
      recon_delta=torch.tensor(residual_error_bound).to(mydevice) # min variation of y^T x /||x||||y||, i.e. y^T x /||x||||y|| > recon_delta
      rho_tensor=torch.tensor(0.).to(mydevice)
   
   # train using buffer 
   for ci in range(n_iter):
      uvw,unscaled_uvw,data,solution,sky,log_eig,residual=buffer.sample_buffer(batch_size,normalize=do_normalize,permute=(np.random.rand() > permute_threshold))
      # no need to clamp (as buffer does the clipping)
      uvw=torch.tensor(uvw).to(mydevice)
      unscaled_uvw=torch.tensor(unscaled_uvw).to(mydevice)
      unscaled_uvw=unscaled_uvw.view(-1,B,3)
      data=torch.tensor(data).to(mydevice)
      solution=torch.tensor(solution).to(mydevice)
      sky=torch.tensor(sky).to(mydevice)
      y=torch.tensor(residual).to(mydevice)
      log_eig=torch.tensor(log_eig).to(mydevice)
      n_batch=uvw.shape[0]
      xtilde, log_px =bd.sample(y,uvw,data,solution,sky,reparameterize=True)

      if not initial_training:
         # log likelihood = log_px - log |J|
         loss1=torch.sum(torch.clamp(log_px,min=-500,max=500))-torch.sum(torch.clamp(log_eig,min=-500,max=500))
         if constraint_type == 0:
             # Inequality constraint, y^T x /||x||||y|| > delta, for delta> 0 (max=1,min=-1)
             # define function g() = max(0,-y^T x/||x||||y|| + delta)^2
             # use lower upper bound to clamp, as final loss scales with gfun**2=upper_bound**4
             # batchwise inner product
             denom=(torch.mean(torch.linalg.norm(y,dim=1))*torch.mean(torch.linalg.norm(xtilde,dim=1))+EPS)
             gfun = torch.sum(torch.clamp(recon_delta-torch.bmm(y.unsqueeze(1),xtilde.unsqueeze(2)).squeeze()/denom,min=0,max=recon_upper_bound).pow(2))
         else : 
             err=xtilde-y
             gfun =torch.sum(torch.clamp(torch.linalg.norm(err,dim=1)**2/(torch.linalg.norm(y,dim=1)**2+EPS) - recon_delta, min=0, max=recon_upper_bound).pow(2))

         loss1 /= n_batch*n_input
         gfun  /= n_batch # do not normalize by n_input
         total_loss=loss1*loss1_scale+0.5*admm_rho*gfun*gfun+rho_tensor*gfun
         eval_log_px,dot_prod=evaluate_model(bd,test_buffer)
         print(f'{total_loss.data.item():.4e} {loss1.data.item():.4e} {gfun.data.item():.4e} {rho_tensor.data.item():.4e} {eval_log_px:.4e} {dot_prod:.4e}')
      else:
         err=xtilde-y
         mse_loss=torch.sum(torch.linalg.norm(err,dim=1)**2)/torch.sum(torch.linalg.norm(y,dim=1)**2+EPS)
         total_loss=mse_loss
         eval_log_px,dot_prod=evaluate_model(bd,test_buffer)
         print(f'{total_loss.data.item():.4e} 0.0 0.0 0.0 {eval_log_px:.4e} {dot_prod:.4e}')

      optimizer.zero_grad()
      total_loss.backward()
      if not initial_training:
         # clip gradient 2 to 5
         nn.utils.clip_grad_norm_(bd.parameters(),max_norm=5,norm_type=2)
      else:
         nn.utils.clip_grad_norm_(bd.parameters(),max_norm=10,norm_type=2)
      optimizer.step()
      scheduler.step()

      # update Lagrange multiplier(s)
      if (ci > 0) and (ci%dual_update_cadence==0):
         with torch.no_grad():
            xtilde, log_px =bd.sample(y,uvw,data,solution,sky,reparameterize=False)
            if not initial_training:
               if constraint_type == 0:
                  denom=(torch.mean(torch.linalg.norm(y,dim=1))*torch.mean(torch.linalg.norm(xtilde,dim=1))+EPS)
                  gfun = torch.sum(torch.clamp(recon_delta-torch.bmm(y.unsqueeze(1),xtilde.unsqueeze(2)).squeeze()/denom,min=0,max=recon_upper_bound).pow(2))
               else:
                  err=xtilde-y
                  gfun =torch.sum(torch.clamp(torch.linalg.norm(err,dim=1)**2/(torch.linalg.norm(y,dim=1)**2+EPS) - recon_delta, min=0, max=recon_upper_bound).pow(2))
               gfun /= n_batch
               rho_tensor += admm_rho*gfun

   if save_model:
      bd.save_checkpoint()
      torch.save(optimizer.state_dict(),'opt.state')

   # evaluate 
   # set scale factors to training levels
   test_buffer.set_normalization(*buffer.calculate_normalization())
   bd.eval() # disable batch normalization and dropout
   Cy=0
   Ci=0
   biasy=0
   biasi=0
   import matplotlib.pyplot as plt
   for ci in range(n_eval):
      uvw,unscaled_uvw,data,solution,sky,log_eig,residual=test_buffer.sample_buffer(1,normalize=do_normalize)
      # no need to clamp all values
      uvw=torch.tensor(uvw).to(mydevice)
      data=torch.tensor(data).to(mydevice)
      solution=torch.tensor(solution).to(mydevice)
      sky=torch.tensor(sky).to(mydevice)
      y=torch.tensor(residual).to(mydevice)
      log_eig=torch.tensor(log_eig).to(mydevice)
      xtilde, log_px =bd.sample(y,uvw,data,solution,sky,reparameterize=False)
      y=y.detach().cpu().numpy().squeeze()
      xtilde=xtilde.detach().cpu().numpy().squeeze()
      Ci=Ci+np.outer(xtilde,xtilde)
      Cy=Cy+np.outer(y,y)
      biasi=biasi+xtilde
      biasy=biasy+y

   Cy/=n_eval
   Ci/=n_eval
   biasi/=n_eval
   biasy/=n_eval
   fig=plt.figure(1,figsize=(15,15))
   fig,axs=plt.subplots(1,2)
   im=axs[0].imshow(np.log(np.abs(Ci))*np.sign(Ci),aspect='auto',interpolation='none',vmin=0,vmax=np.log(np.abs(Ci).max()))
   axs[0].set_xlabel('P(x)')
   plt.colorbar(im,ax=axs[0])
   im=axs[1].imshow(np.log(np.abs(Cy))*np.sign(Cy),aspect='auto',interpolation='none',vmin=0,vmax=np.log(np.abs(Cy).max()))
   axs[1].set_xlabel('Residual')
   plt.colorbar(im,ax=axs[1])
   plt.savefig('cov_'+str(epoch)+'.png',dpi=300)

   plt.clf()
   # calculate uvw distance to sort plot
   uvw=uvw.reshape(B,3)
   uvwd=torch.sqrt(uvw[:,0]**2+uvw[:,1]**2)
   uvw=torch.zeros(uvw.shape[0],8)
   for col in range(8):
      uvw[:,col]=uvwd
   sorted_idx=torch.sort(uvw.flatten()).indices.cpu().numpy()
   fig,axs=plt.subplots()
   axs.plot(biasi[sorted_idx],label=f'P(x)')
   axs.plot(biasy[sorted_idx],label=f'Residual')
   axs.legend()
   plt.savefig('mean_'+str(epoch)+'.png',dpi=300)
