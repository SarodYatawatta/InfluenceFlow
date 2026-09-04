#! /usr/bin/env python
import argparse
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.functional
import torch.optim as optim
from elasticnet_model import ElasticNetModel
from dnn_models import BaseDist
from training_buffer import TrainingBuffer
from autograd_tools import *

###
# Use cross correlation between x ~ p(x) and y = residual
# as an inequality constraint x^T y / ||x||||y|| > min_bound
###
# (try to) use a GPU for computation?
use_cuda=True
if use_cuda and torch.cuda.is_available():
  mydevice=torch.device('cuda')
else:
  mydevice=torch.device('cpu')

# epsilon (used in various places)
EPS=1e-5

N=16 # input, noise will be sqrt(N)xsqrt(N)
M=14
# training iterations
n_iter=250000
# evaluation iterations
n_eval=3000
batch_size=256
n_buffer=40000
load_buffer=0 # if 1, load, if 0, simulate and save

random_seed=1

log_eig_eps=EPS
admm_rho=1e-3 # start with a low value, and increase
residual_error_bound=0.8 # delta, max value is 1 = y^T x /||y||||x|| per each batch (normalized by input) 
recon_upper_bound=2 # upper bound of (-y^T x /||y||||x|| +delta), per each batch
dual_update_cadence=100 # how many iterations for updating Lagrange multiplier 

# loss normalization factors
loss1_scale=-1 # maximize loss

# hidden layer dim
n_hidden=256
# depth of layers
# Note: highly sensitive to depth, 4 or 5 chosen
n_depth=5

# learning rate: start with 0 -> target_lr, thereafter
# switch to cosine annealing till the end
target_lr=1e-4
# where to end warmup, look at wild swings in cost and decide
warmup_steps=5000

initial_training=0
load_model=0
save_model=1
do_normalize=False
dropout=False

parser = argparse.ArgumentParser(description="Train normalizing flow",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def init_parser():
    parser.add_argument('--model_N', default=N, type=int, help='Number of input/output variables, if perfect square, Gaussian random field noise')
    parser.add_argument('--model_M', default=M, type=int, help='Number of parameters')
    parser.add_argument('--iterations', default=n_iter, type=int, help='training iterations')
    parser.add_argument('--warmup', default=warmup_steps, type=int, help='warmup iterations')
    parser.add_argument('--evaluations', default=n_eval, type=int, help='evaluation iterations')
    parser.add_argument('--batch_size', default=batch_size, type=int, help='batch size')
    parser.add_argument('--learning_rate', default=target_lr, type=float, help='learning rate')
    parser.add_argument('--seed', default=random_seed, type=int, help='random seed')
    parser.add_argument('--data_size', default=n_buffer, type=int, help='training data size')
    parser.add_argument('--load_data', default=load_buffer, type=int, help='load saved data buffer (if 0, data will be generated)')
    parser.add_argument('--load_model', default=load_model, type=int, help='load saved model')
    parser.add_argument('--save_model', default=save_model, type=int, help='save model')
    parser.add_argument('--regularization', default=admm_rho, type=float, help='regularization factor for correlation constraint')
    parser.add_argument('--bound', default=residual_error_bound, type=float, help='lower bound of the constraint')
    parser.add_argument('--cadence', default=dual_update_cadence, type=int, help='cadence to update dual (Lagrange multiplier)')
    parser.add_argument('--DNN_hidden', default=n_hidden, type=int, help='DNN hidden layer dimension')
    parser.add_argument('--DNN_depth', default=n_depth, type=int, help='DNN layer depth')
    parser.add_argument('--initial_training', default=initial_training, type=int, help='If 1, train the model to predict the residual')


init_parser()
args = parser.parse_args()

# get back command line args
M=args.model_M
N=args.model_N
n_iter=args.iterations
n_eval=args.evaluations
batch_size=args.batch_size
n_buffer=args.data_size
target_lr=args.learning_rate
warmup_steps=args.warmup
random_seed=args.seed
load_buffer=args.load_data
admm_rho=args.regularization
residual_error_bound=args.bound
dual_update_cadence=args.cadence
save_model=args.save_model
load_model=args.load_model
n_hidden=args.DNN_hidden
n_depth=args.DNN_depth
initial_training=args.initial_training

n_input=N # ideally sqrt(N)xsqrt(N) square

def evaluate_model(model,test_buffer):
   n_m_eval=min(256,n_eval)
   sum_log_px=0
   dot_prod=0
   model.eval()
   with torch.no_grad():
      y,log_eig,J,metadata=test_buffer.sample_buffer(n_m_eval,normalize=do_normalize)
      y=torch.tensor(y).to(mydevice)
      log_eig=torch.tensor(log_eig).to(mydevice)
      metadata=torch.tensor(metadata).to(mydevice)
      xtilde,log_px,_,_ =model.sample(y,metadata,reparameterize=False)
      sum_log_px +=torch.sum(log_px,dim=1)
      # correlation or cosine similarity
      dot_prod += torch.sum(torch.bmm(y.unsqueeze(1),xtilde.unsqueeze(2)).squeeze())/(torch.mean(torch.linalg.norm(y,dim=1))*torch.mean(torch.linalg.norm(xtilde,dim=1))+EPS)

   model.train()
   return torch.sum(sum_log_px).cpu().numpy()/(n_m_eval*n_input),dot_prod/n_m_eval

for epoch in range(1):
   enm=ElasticNetModel(N=N,M=M,seed=random_seed+epoch,SNR=1)
   # increasing depth > 4 lose covariance diagonal
   bd=BaseDist(n_input,n_metadata=enm.metadata_size,n_hidden=n_hidden,depth=n_depth,epsilon=0.01,dropout=dropout)
   if load_model:
       bd.load_checkpoint()
   total_parameters=bd.parameters()
   buffer=TrainingBuffer(n_buffer,n_input,enm.metadata_size)
   test_buffer=TrainingBuffer(n_eval,n_input,enm.metadata_size)
   if not load_buffer:
      # fill buffer
      for cj in range(n_buffer):
         enm.reset()
         enm.run()
         enm.fit()
         enm.calculate_influence()
         y=enm.residual().numpy()
         J=enm.F
         eigs,_=torch.linalg.eig(J)
         # count negative eigs
         eigs=torch.real(eigs)
         neg_eigs=torch.sum(eigs<EPS)
         # eigs > 0 for possible invertible map, so set all -ve eigenvalues to ~0
         eigs[eigs<EPS]=EPS
         # normalize (after finding eigenvalues)
         J=J/torch.linalg.matrix_norm(J)
         J=J.numpy()
         metadata=enm.metadata().detach().cpu().numpy()
         log_eig=torch.log(eigs+log_eig_eps).numpy()
         # make sure design matrix has full rank, no check done here
         buffer.store_observation(y,log_eig,J,metadata)
      buffer.save_checkpoint('buffer_'+str(epoch)+'.npy')
      for cj in range(n_eval):
         enm.reset()
         enm.run()
         enm.fit()
         enm.calculate_influence()
         y=enm.residual().numpy()
         J=enm.F
         eigs,_=torch.linalg.eig(J)
         eigs=torch.real(eigs)
         eigs[eigs<EPS]=EPS
         # normalize (after finding eigenvalues)
         J=J/torch.linalg.matrix_norm(J)
         J=J.numpy()
         metadata=enm.metadata().detach().cpu().numpy()
         log_eig=torch.log(eigs+log_eig_eps).numpy()
         test_buffer.store_observation(y,log_eig,J,metadata)
      test_buffer.save_checkpoint('test_buffer_'+str(epoch)+'.npy')
   else:
      # load saved buffer
      buffer.load_checkpoint('buffer_'+str(epoch)+'.npy')
      test_buffer.load_checkpoint('test_buffer_'+str(epoch)+'.npy')

   # copy normalization
   if do_normalize:
      test_buffer.set_normalization(*buffer.calculate_normalization())

   #optimizer=optim.Adam(total_parameters,lr=target_lr)
   # AdamW is more stable for long runs
   optimizer=optim.AdamW(total_parameters,lr=target_lr,weight_decay=1e-5)

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
      y,log_eig,J,metadata=buffer.sample_buffer(batch_size,normalize=do_normalize)
      y=torch.tensor(y).to(mydevice)
      log_eig=torch.tensor(log_eig).to(mydevice)
      metadata=torch.tensor(metadata).to(mydevice)
      n_batch=J.shape[0]
      xtilde, log_px, _, _=bd.sample(y,metadata,reparameterize=True)

      if not initial_training:
         # log likelihood = log_px - log |J|
         # NB: make sure to adjust clamping values to each problem
         loss1=torch.sum(torch.clamp(log_px,min=-500*n_input,max=500*n_input))-torch.sum(torch.clamp(log_eig,min=-100*n_input,max=100*n_input))
         # Inequality constraint, y^T x /E{||y|}E{|||x||} > delta (max=1, min=-1)
         # or,  - y^T x/||y||||x|| < -delta 
         # define function g() = max(0,-y^T x//||x||||y|| + delta)^2
         # use lower upper bound to clamp, as final loss scales with gfun**2=upper_bound**4
         denom=(torch.mean(torch.linalg.norm(y,dim=1))*torch.mean(torch.linalg.norm(xtilde,dim=1))+EPS)
         gfun = torch.sum(torch.clamp(recon_delta-torch.bmm(y.unsqueeze(1),xtilde.unsqueeze(2)).squeeze()/denom,min=0,max=recon_upper_bound).pow(2))
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
         # clip gradient - 2 to 5
         nn.utils.clip_grad_norm_(bd.parameters(),max_norm=5,norm_type=2)
      else:
         nn.utils.clip_grad_norm_(bd.parameters(),max_norm=10,norm_type=2)
      optimizer.step()
      scheduler.step()

      # update Lagrange multiplier
      if (not initial_training) and ci>0 and (ci%dual_update_cadence==0):
         with torch.no_grad():
            xtilde, log_px, _, _=bd.sample(y,metadata,reparameterize=False)
            denom=(torch.mean(torch.linalg.norm(y,dim=1))*torch.mean(torch.linalg.norm(xtilde,dim=1))+EPS)
            gfun = torch.sum(torch.clamp(recon_delta-torch.bmm(y.unsqueeze(1),xtilde.unsqueeze(2)).squeeze()/denom,min=0,max=recon_upper_bound).pow(2))
            gfun /= n_batch
            rho_tensor += admm_rho*gfun

   if save_model:
      bd.save_checkpoint()

   # final evaluation
   Cy=0
   Ci=0
   biasy=0
   biasi=0
   bd.eval()
   import matplotlib.pyplot as plt
   for ci in range(n_eval):
      y,log_eig,J,metadata=test_buffer.sample_buffer(1,normalize=do_normalize)
      y=torch.tensor(y).to(mydevice)
      log_eig=torch.tensor(log_eig).to(mydevice)
      metadata=torch.tensor(metadata).to(mydevice)
      xtilde,_,_,_=bd.sample(y,metadata)
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
   fig=plt.figure(1)
   fig,axs=plt.subplots(1,3)
   target=enm.u_C
   C=target.detach().cpu().numpy()
   im=axs[0].imshow(C,aspect='auto')#,vmin=0,vmax=C.max())
   axs[0].set_xlabel('Model')
   axs[1].imshow(Ci,aspect='auto')#,vmin=0,vmax=C.max())
   axs[1].set_xlabel('P(x)')
   axs[2].imshow(Cy,aspect='auto')#,vmin=0,vmax=C.max())
   axs[2].set_xlabel('Residual')
   plt.colorbar(im,ax=axs[2])
   plt.savefig('cov_'+str(epoch)+'.png')

   plt.clf()
   fig,axs=plt.subplots()
   b_mu=enm.u_b.detach().cpu().numpy()
   if do_normalize:
      r_mu,r_std,_,_=buffer.calculate_normalization()
      # map back to data space
      biasi=biasi*r_std+r_mu
      biasy=biasy*r_std+r_mu
   # normalize by 1/||ground_truth||^2
   b_dot_biasi=np.dot(b_mu,biasi)/(np.linalg.norm(b_mu)**2)
   b_dot_biasi_norm=np.dot(b_mu,biasi)/(np.linalg.norm(b_mu)*np.linalg.norm(biasi))
   b_dot_res=np.dot(b_mu,biasy)/(np.linalg.norm(b_mu)**2)
   b_dot_res_norm=np.dot(b_mu,biasy)/(np.linalg.norm(b_mu)*np.linalg.norm(biasy))
   axs.plot(np.arange(biasi.size),b_mu,'-o',label=f'Model, norm {np.linalg.norm(b_mu):.2e}')
   axs.plot(biasi,'-*',label=f'P(x), corr {b_dot_biasi:.2e}, {b_dot_biasi_norm:.2e}')
   axs.plot(biasy,'-+',label=f'Residual, corr {b_dot_res:.2e}, {b_dot_res_norm:.2e}')
   axs.legend()
   plt.savefig('mean_'+str(epoch)+'.png')
   plt.close()
