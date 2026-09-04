#! /usr/bin/env python
import argparse
import numpy as np
import torch
import generate_data
import casa_io
import calibration_tools
from training_buffer import TrainingBuffer
from dnn_models import BaseDist

# (try to) use a GPU for computation?
use_cuda=True
if use_cuda and torch.cuda.is_available():
  mydevice=torch.device('cuda')
else:
  mydevice=torch.device('cpu')


K=6
n_sol=300
n_sol_interval=10

n_buffer=15000
# hidden layer dim ~ B
n_hidden=2048
# depth of layers (total depth x2 + 1 this value)
n_depth=1
error_threshold=1e-1
frequency0=150e6

do_batchnorm=0 # if 1, enable batchnorm
load_model=1
do_normalize=False
dropout=False

parser = argparse.ArgumentParser(description="Evaluate trained normalizing flow with one observation",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def init_parser():
    parser.add_argument('--K', default=K, type=int, help='Number of directions')
    parser.add_argument('--data_size', default=n_buffer, type=int, help='Training data size')
    parser.add_argument('--load_model', default=load_model, type=int, help='Load saved model')
    parser.add_argument('--DNN_hidden', default=n_hidden, type=int, help='DNN hidden layer dimension')
    parser.add_argument('--DNN_depth', default=n_depth, type=int, help='DNN layer depth')
    parser.add_argument('--batchnorm', default=do_batchnorm, type=int, help='If 1, enable batchnorm')
    parser.add_argument('--num_sol', default=n_sol, type=int, help='Number of solutions to simulate')
    parser.add_argument('--num_sol_interval', default=n_sol_interval, type=int, help='Number of time samples per solution (1 sec)')
    parser.add_argument('--threshold', default=error_threshold, type=float, help='Resudial error threshold to flag data')
    parser.add_argument('--frequency', default=frequency0, type=float, help='Frequency of simulation')
    parser.add_argument('--seed',type=int,default=123,help='Initial random seed')

init_parser()
args = parser.parse_args()

# get back command line args
K=args.K
n_buffer=args.data_size
load_model=args.load_model
n_hidden=args.DNN_hidden
n_depth=args.DNN_depth
do_batchnorm=args.batchnorm
n_sol=args.num_sol
n_sol_interval=args.num_sol_interval
error_threshold=args.threshold
frequency0=args.frequency

n_stat,uvw,data,sol,sky,log_eig,residual,msname=\
        generate_data.generate_training_data(K,n_sol=n_sol,n_sol_interval=n_sol_interval,t_integration=10,seed=args.seed,do_influence=False,frequency=frequency0)

del uvw,data,sol,log_eig,residual

N=n_stat
# corr_data: not used (influence function not calculated)
n_stat,uvw,data,corr_data,model_data=casa_io.read_corr_full(msname,full_time=True)
freq0,J=calibration_tools.readsolutions('zsol')
B=n_stat*(n_stat-1)//2
# input dimensions: 8xbaselines, fixed
n_input=B*8

# check shapes
assert(J.shape==(K,2*n_stat*n_sol,2))
assert(uvw.shape==(B*n_sol*n_sol_interval,3))
assert(data.shape==(B*n_sol*n_sol_interval,8))
assert(corr_data.shape==(B*n_sol*n_sol_interval,8))
assert(model_data.shape==(B*n_sol*n_sol_interval,8))

buffer=TrainingBuffer(n_buffer,N,K)
buffer.load_checkpoint('buffer_0.npy')
if do_normalize:
   uvw_mu_,uvw_std_,data_mu_,data_std_,sol_mu_,sol_std_,\
      sky_mu_,sky_std_,res_mu_,res_std_=buffer.calculate_normalization()
else:
   uvw_mu_=data_mu_=sol_mu_=sky_mu_=res_mu_=0
   uvw_std_=data_std_=sol_std_=sky_std_=res_std_=1

# scale factors (product of simulation scale factor and scale factor applied to buffer)
uvw_scale=(1/1000)
data_scale=100
solution_scale=1
sky_scale=1
residual_scale=100
print(f'Scale factors : uvw {uvw_scale:.4e} data {data_scale:.4e} solution {solution_scale:.4e} sky {sky_scale:.4e} residual {residual_scale:.4e}')

bd=BaseDist(N,K,n_hidden=n_hidden,depth=n_depth,epsilon=1e-4,batchnorm=do_batchnorm,dropout=dropout)
if load_model:
   bd.load_checkpoint()
bd=torch.compile(bd)
bd.eval()
# enable dropout
if dropout:
   for m in bd.modules():
      if isinstance(m,torch.nn.Dropout):
        m.train()

   # Number of Monte Carlo runs
   n_samples=150
   n_elites=40
   n_cem=5
else:
   n_samples=5
   n_elites=1
   n_cem=1

CLIP=1000
def convert_uvw(uvw):
    uvw=uvw.reshape(B,3)
    uvw_amp=np.sqrt(uvw[:,0]**2+uvw[:,1]**2+uvw[:,2]**2)
    uvw[:,0] *=np.log(uvw_amp)/uvw_amp
    uvw[:,1] *=np.log(uvw_amp)/uvw_amp
    uvw[:,2] *=np.log(uvw_amp)/uvw_amp
    return uvw.flatten()

def cem_optimize(model,y_b,uvw_b,data_b,solution_b,sky_b):
    var_threshold=0.5
    xtilde=y_b.squeeze()
    xvars=torch.zeros(n_cem)
    flag_mask=torch.zeros(1).to(mydevice)
    for ct in range(n_cem):
       xbatch=[]
       for ci in range(n_samples):
          x, _=model.sample(y_b,uvw_b,data_b,solution_b,sky_b)
          xbatch.append(x.squeeze())
       xbatch=torch.stack(xbatch)
       xvar=torch.var(xbatch,dim=0)
       flag_mask=torch.logical_or((xvar > var_threshold), flag_mask)
       xvars[ct]=torch.mean(xvar)
       xerr=xbatch-xtilde
       scores=torch.linalg.norm(xerr,dim=1)
       elite_indices=torch.argsort(scores)[:n_elites]
       elites=xbatch[elite_indices]
       xtilde=torch.mean(elites,dim=0)
    flag_ratio=torch.sum(flag_mask)/(flag_mask.size()[0])
    xtilde[flag_mask]=0
    return xtilde, torch.mean(xvars), flag_ratio

# as the data is not phase shifted, all images made at center 0,pi/2
generate_data.make_image_col(msname,'DATA','data',ra=0,dec=np.pi/2,fullpol=True)
generate_data.make_image_col(msname,'MODEL_DATA','res',ra=0,dec=np.pi/2,fullpol=True)

c_const=299792458.0
uvw *=(freq0/c_const) # wavelengths
uvw *=uvw_scale
data *=data_scale
J *=solution_scale
sky *=sky_scale
model_data *=residual_scale

prediction=data.copy()
sky=torch.tensor(sky.flatten())
# iterate over all time slots (fill one batch before eval of model)
n_filled=0
out_counter=0

for t in range(n_sol*n_sol_interval):
    # solution index
    s=t//n_sol_interval
    Js=J[:,2*n_stat*s:2*n_stat*s+2*n_stat,:]
    if t==0:
        J0=Js.copy()
    else:
        for ndir in range(K):
          J1J=np.matmul(np.matrix(Js[ndir]).H,np.matrix(J0[ndir]))
          uu,ss,vvh=np.linalg.svd(J1J)
          uu1=np.matmul(uu,vvh)
          Js[ndir]=np.matmul(Js[ndir],uu1)

    Jsol=np.concatenate((Js.real,Js.imag)).flatten()
    uvw_s=uvw[t*B:t*B+B,:].flatten()
    # convert uvw to log scale (scaling done twice)
    uvw_s=convert_uvw(uvw_s)
    uvw_s =(uvw_s-uvw_mu_)/uvw_std_
    data_s=data[t*B:t*B+B,:].flatten()
    model_data_s=model_data[t*B:t*B+B,:].flatten()
    data_s=(data_s-data_mu_)/data_std_
    model_data_s=(model_data_s-res_mu_)/res_std_
    uvw_t=torch.tensor(uvw_s)
    data_t=torch.tensor(data_s)
    solution_t=torch.tensor(Jsol)
    y=torch.tensor(model_data_s)
    y_b=y.to(mydevice).unsqueeze(0)
    uvw_b=uvw_t.to(mydevice).unsqueeze(0)
    data_b=data_t.to(mydevice).unsqueeze(0)
    solution_b=solution_t.to(mydevice).unsqueeze(0)
    sky_b=sky.to(mydevice).unsqueeze(0)
    xtilde,xvar,flag_ratio=cem_optimize(bd,y_b,uvw_b,data_b,solution_b,sky_b)
    mse_loss=torch.sum(torch.linalg.norm(xtilde-y_b,dim=1)**2)/torch.sum(torch.linalg.norm(y_b,dim=1)**2)
    print(f'{t} {mse_loss.data.item():.4e} {xvar.data.item():.4e} {flag_ratio.data.item():.2e}')
    if mse_loss > error_threshold:
             xtilde=torch.zeros_like(xtilde).cpu().numpy().squeeze()
    else:
             xtilde=xtilde.detach().cpu().numpy().squeeze()
             xtilde = xtilde*res_std_+res_mu_
             xtilde *=1/(residual_scale)
             xtilde=np.clip(xtilde,-CLIP,CLIP)
    prediction[t*B:t*B+B,:]=xtilde.reshape(B,8)

# total error
mse_loss=np.linalg.norm(model_data.flatten()-prediction.flatten())**2/np.linalg.norm(model_data.flatten())**2
print(f'final {mse_loss:.4e}')
# write prediction to MS and image
casa_io.write_corr_full(msname,'CORRECTED_DATA',prediction.flatten(),full_time=True)
# make diffuse model image (last step) DATA will be overwritten
_,_,_,_=generate_data.make_model_image(msname,0,np.pi/2,K,tslots=n_sol_interval)
generate_data.make_image_col(msname,'CORRECTED_DATA','predict',ra=0,dec=np.pi/2,fullpol=True)
generate_data.make_image_col(msname,'DATA','model',ra=0,dec=np.pi/2,fullpol=True)
