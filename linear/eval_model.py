#! /usr/bin/env python
import argparse
import numpy as np
import torch
import torch.nn as nn
from elasticnet_model import ElasticNetModel
from training_buffer import TrainingBuffer
from dnn_models import BaseDist
# for visualizing model
from torchview import draw_graph
# for saving .mat
from scipy.io import savemat

# (try to) use a GPU for computation?
use_cuda=True
if use_cuda and torch.cuda.is_available():
  mydevice=torch.device('cuda')
else:
  mydevice=torch.device('cpu')

N=15
M=12
n_eval=10

random_seed=1

  
# hidden layer dim
n_hidden=256
n_depth=5

n_buffer=100
do_normalize=False
dropout=False

parser=argparse.ArgumentParser(description="Evaluate normalizing flow",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
def init_parser():
    parser.add_argument('--model_N', default=N, type=int, help='Number of input/output variables, if perfect square, Gaussian random field noise')
    parser.add_argument('--model_M', default=M, type=int, help='Number of parameters')
    parser.add_argument('--evaluations', default=n_eval, type=int, help='evaluation iterations')
    parser.add_argument('--seed', default=random_seed, type=int, help='random seed')

    parser.add_argument('--data_size', default=n_buffer, type=int, help='training data size')
    parser.add_argument('--DNN_hidden', default=n_hidden, type=int, help='DNN hidden layer dimension')
    parser.add_argument('--DNN_depth', default=n_depth, type=int, help='DNN layer depth')

init_parser()
args=parser.parse_args()
M=args.model_M
N=args.model_N
n_eval=args.evaluations
random_seed=args.seed
n_buffer=args.data_size
n_hidden=args.DNN_hidden
n_depth=args.DNN_depth

n_input=N # ideally, sqrt(N)*sqrt(N)
Ns=int(np.sqrt(N))

enm=ElasticNetModel(N=N,M=M,seed=random_seed,SNR=1)
bd=BaseDist(n_input,n_metadata=enm.metadata_size,n_hidden=n_hidden,depth=n_depth,epsilon=0.01,dropout=dropout)

bd.load_checkpoint()
bd.eval()

if dropout:
   # enable dropout
   for m in bd.modules():
       if isinstance(m,torch.nn.Dropout):
           m.train()
# how many samples to draw for each eval (only when dropout enabled)
n_samples=1

buffer=TrainingBuffer(n_buffer,n_input,enm.metadata_size)
buffer.load_checkpoint('buffer_0.npy')
if do_normalize:
   r_mu,r_std,metadata_mu,metadata_std=buffer.calculate_normalization()
   r_mu=torch.tensor(r_mu).to(mydevice)
   r_std=torch.tensor(r_std).to(mydevice)
   metadata_mu=torch.tensor(metadata_mu).to(mydevice)
   metadata_std=torch.tensor(metadata_std).to(mydevice)
else:
   r_mu=0
   r_std=1
   metadata_mu=0
   metadata_std=1

########################################
draw_model=False
if draw_model:
   model_graph = draw_graph(
    bd,
    input_size=[(1, n_input),(1,enm.metadata_size)],
    graph_name="model_architecture",
    depth=2,                        # Control layer detail nesting level
    expand_nested=True,             # Set False for high-level block diagram
    graph_dir="TB",                 # orientation
    roll=True                       # Rolls recurrent/loop layers to clean up the graph
   )

   model_graph.visual_graph.attr(dpi="300")               # Match journal resolution minima
   model_graph.visual_graph.attr('node', fontname="Arial") # Use clean academic fonts
   model_graph.visual_graph.attr('edge', fontname="Arial")

   model_graph.visual_graph.render(filename="network_figure", format="pdf", cleanup=True)
########################################
Cy=0
Ci=0
biasy=0
biasi=0
datai=0
for ci in range(n_eval):
   enm.reset()
   enm.run()
   enm.fit()
   enm.calculate_influence()
   y=(enm.residual().to(mydevice)-r_mu)/r_std
   metadata=(enm.metadata().detach().to(mydevice)-metadata_mu)/metadata_std
   y=y[None,]
   metadata=metadata[None,]
   xbatch=0
   with torch.no_grad():
     for ns in range(n_samples):
        xtilde, log_px, _, _=bd.sample(y,metadata,reparameterize=False)
        xbatch+=xtilde.detach().cpu().numpy().squeeze()
   xbatch /= n_samples
   y=y.detach().cpu().numpy().squeeze()
   Ci=Ci+np.outer(xbatch,xbatch)
   Cy=Cy+np.outer(y,y)
   biasi=biasi+xbatch
   biasy=biasy+y
   datai=datai+((enm.y-r_mu)/r_std).detach().cpu().numpy()

Cy/=n_eval
Ci/=n_eval
# mean
biasi/=n_eval
biasy/=n_eval
datai/=n_eval

import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "serif",          # Use "sans-serif" if preferred
    "font.serif": ["Times New Roman", "DejaVu Serif"], # Fallback order
    "font.size": 10,                 # Standard journal caption/body size
    "axes.labelsize": 11,            # Font size of the x and y labels
    "axes.titlesize": 12,            # Font size of the plot title
    "xtick.labelsize": 9,            # Font size of the x-axis tick labels
    "ytick.labelsize": 9,            # Font size of the y-axis tick labels
    "mathtext.fontset": "stix"       # Makes LaTeX math text match Times New Roman
})

fig=plt.figure(1)
fig,axs=plt.subplots(1,3)
target=enm.u_C
C=target.detach().cpu().numpy()
im=axs[0].imshow(C,aspect='auto')#,vmin=0,vmax=C.max())
axs[0].set_xlabel('Model')
axs[2].imshow(Cy,aspect='auto')#,vmin=0,vmax=C.max())
axs[2].set_xlabel('Residual')
axs[1].imshow(Ci,aspect='auto')#,vmin=0,vmax=C.max())
axs[1].set_xlabel('P(x)')
plt.colorbar(im,ax=axs[2])
plt.savefig('corr.png')

x=np.arange(biasi.size)
b_mu=enm.u_b.detach().cpu().numpy()
# map back to data space
if do_normalize:
   r_mu=r_mu.detach().cpu().numpy()
   r_std=r_std.detach().cpu().numpy()
   biasi=biasi*r_std+r_mu
   biasy=biasy*r_std+r_mu
   datai=datai*r_std+r_mu

# normalize by 1/||ground_truth||^2
b_dot_biasi=np.dot(b_mu,biasi)/(np.linalg.norm(b_mu)**2)
b_dot_res=np.dot(b_mu,biasy)/(np.linalg.norm(b_mu)**2)
# perfect square case 
if N==Ns*Ns:
   plt.clf()
   fig,axs=plt.subplots(2,2)
   # use same limits
   min_range=-0.05
   max_range=0.05
   im=axs[0,1].imshow(b_mu.reshape(Ns,Ns),aspect='auto',vmin=min_range,vmax=max_range)
   axs[0,1].set_xlabel('Model')
   plt.colorbar(im,ax=axs[0,1],shrink=0.6)
   im=axs[1,1].imshow(biasi.reshape(Ns,Ns),aspect='auto',vmin=min_range,vmax=max_range)
   axs[1,1].set_xlabel(f'P(x) {b_dot_biasi:.4f}')
   plt.colorbar(im,ax=axs[1,1],shrink=0.6)
   im=axs[1,0].imshow(biasy.reshape(Ns,Ns),aspect='auto',vmin=min_range,vmax=max_range)
   axs[1,0].set_xlabel(f'Residual {b_dot_res:.4f}')
   plt.colorbar(im,ax=axs[1,0],shrink=0.6)
   im=axs[0,0].imshow(datai.reshape(Ns,Ns),aspect='auto',vmin=min_range,vmax=max_range)
   axs[0,0].set_xlabel(f'Observed')
   plt.colorbar(im,ax=axs[0,0],shrink=0.6)
   plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2.0, w_pad=2.0)
   plt.savefig('mean.png')
else:
   plt.clf()
   fig,axs=plt.subplots()
   axs.plot(np.arange(biasi.size),b_mu,'-o',label=f'Model, norm {np.linalg.norm(b_mu):.2e}')
   axs.plot(biasi,'-*',label=f'P(x), corr {b_dot_biasi:.2e}')
   axs.plot(biasy,'-+',label=f'Residual, corr {b_dot_res:.2e}')
   axs.plot(datai,'-',label=f'Data')
   axs.legend()
   plt.savefig('mean.png')
mydict={'bmu': b_mu, 'biasi': biasi, 'biasy': biasy, 'datai': datai}
savemat('aa.mat',mydict)
plt.close()
