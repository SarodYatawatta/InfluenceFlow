import numpy as np
import torch
from lbfgsnew import LBFGSNew
from autograd_tools import *
from covariance_generator import covariance_generator
from grf import generate_2d_grf


# (try to) use a GPU for computation?
use_cuda=True
if use_cuda and torch.cuda.is_available():
  mydevice=torch.device('cuda')
else:
  mydevice=torch.device('cpu')


class ElasticNetModel:
    # solve
    # arg min_x || y - Ax ||^2 + \lambda1 (||x||_2)^2 + \lambda2 ||x||_1
    # A: tall matrix NxM, N>M
    def __init__(self,N=9,M=5,lambda1=0.001,lambda2=0.001,SNR=10,seed=None):
        self.N=N
        self.M=M
        self.lambda1=lambda1
        self.lambda2=lambda2
        self.SNR=SNR
        # random seed (for base dist)
        self.seed=seed
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        # create fixed base distribution parameters
        # (root) covariance (use Cholesky factor)
        C=covariance_generator(self.N)
        self.u_C=C.to(mydevice)
        self.u_Cf=torch.linalg.cholesky(self.u_C)
        Ns=int(np.sqrt(N))
        if (N==Ns*Ns):
            self.is_sqr=True
            # bias: Gaussian random field of sqrt(N) x sqrt(N)
            bias=generate_2d_grf(size=Ns,seed=22,Hurst=2).flatten()
            self.u_b=torch.FloatTensor(bias).to(mydevice)*0.02
        else:
            self.is_sqr=False
            # bias: Gaussian vector
            self.u_b=torch.randn(self.N).to(mydevice)*0.1
        self.A=None
        self.x0=None
        self.y0=None
        self.y=None
        self.n=None
        self.x=None
        self.F=None
        self.opt=None

        # metadata: A, x
        self.metadata_size=self.N*self.M+self.M

    def reset(self):
        # create a new simulation (matrix can lose rank)
        self.A=torch.randn(self.N,self.M,dtype=torch.float32,requires_grad=False,device=mydevice)#+np.sqrt(self.N)*torch.eye(self.N,self.M,dtype=torch.float32,requires_grad=False,device=mydevice)
        # use U[0,1] for parameters so that E{x} != 0
        self.x0=torch.rand(self.M,dtype=torch.float32,requires_grad=False,device=mydevice)
        self.y0=torch.matmul(self.A,self.x0)

    def run(self):
        # run the simulation (add noise from base distribution)
        n=torch.randn(self.N,dtype=torch.float32,requires_grad=False,device=mydevice)
        # generate noise from base distribution: scale such that
        # adding about 400 realizations would show u_b
        if self.is_sqr:
           self.n=torch.matmul(self.u_Cf,n)/np.sqrt(self.N)+self.u_b
        else:
           self.n=torch.matmul(self.u_Cf,n)+self.u_b
        # scale input to get SNR, keep noise power same
        self.y=self.SNR*torch.norm(self.n)/torch.norm(self.y0)*self.y0+self.n

    def lossfunction(self,A,y,x,alpha=0.0,beta=0.0):
            Ax=torch.matmul(A,x)
            err=y-Ax
            return torch.norm(err,2)**2+alpha*torch.norm(x,2)**2+beta*torch.norm(x,1)

    def fit(self):
        # solve for the unknowns
        # parameters, initialized to zero
        self.x=torch.zeros(self.M,requires_grad=True,device=mydevice)

        self.opt = LBFGSNew([self.x],history_size=7,max_iter=10,line_search_fn=True,batch_mode=False)

        # find solution x
        for nepoch in  range(0,20):
             def closure():
                if torch.is_grad_enabled():
                   self.opt.zero_grad()
                loss=self.lossfunction(self.A,self.y,self.x,self.lambda1,self.lambda2)
                if loss.requires_grad:
                   loss.backward()
                return loss
  
             self.opt.step(closure)

    def calculate_influence(self):
        # find Jacobian of the transform
        # find Jacobian of model = A
        jac=jacobian(torch.matmul(self.A,self.x),self.x).to(mydevice)
        # right hand term = -2 A^T
        df_dx=(lambda yi: gradient(self.lossfunction(self.A,yi,self.x,self.lambda1,self.lambda2), self.x))
        # no need to pass one-hot vectors, because we calculate d( )/dy^T in one go
        e=torch.ones_like(self.y) # all ones
        ll=torch.autograd.functional.jacobian(df_dx,e)

        mm=torch.zeros_like(ll).to(mydevice)
        # copy ll because it is modified
        for i in range(self.N):
            ll2=ll[:,i].clone().detach()
            mm[:,i]=inv_hessian_mult(self.opt,ll2)

        # multiply by Jacobian of model, add I
        # y = x - f, so J = I - \cal A
        self.F=torch.eye(self.N)+torch.matmul(jac,mm).detach().to('cpu')
        # ideally this should be normalized, only after finding eigenvalues

        # check
        #AtAi=torch.linalg.pinv(torch.matmul(self.A.t(),self.A))
        #Atilde=-torch.matmul(self.A,torch.matmul(AtAi,self.A.t()))
        #print('A')
        #print(torch.eye(self.N).to(mydevice)-Atilde)
        #print('F')
        #print(self.F)

        # eigenvalues
        #E,_=torch.linalg.eig(self.F)

    def residual(self):
        # calculate residual
        r=self.y - torch.matmul(self.A,self.x)
        return r.detach().cpu()

    def metadata(self):
        # return metadata (normalized) as a vector
        return torch.cat((torch.flatten(self.A)/torch.linalg.norm(self.A),self.x/torch.linalg.norm(self.x)))

if 0:
   import matplotlib.pyplot as plt
   fig=plt.figure(1)
   fig,axs=plt.subplots(2,2)
   N=10
   enm=ElasticNetModel(N=N*N,M=2)
   n_simul=1
   n_runs=100
   for ci in range(n_simul):
      enm.reset()
      C=0
      m=0
      for cj in range(n_runs):
         enm.run()
         enm.fit()
         enm.calculate_influence()
         r=enm.residual().numpy()
         m+=r
         C=C+np.outer(r,r)
      C/=n_runs
      m/=n_runs
      m=m.reshape(N,N)
      target=enm.u_C
      axs[0,0].imshow(target.detach().cpu().numpy(),aspect='auto')
      axs[0,1].imshow(C,aspect='auto')
      axs[1,0].imshow(enm.u_b.view(N,N).detach().cpu().numpy(),aspect='auto')
      axs[1,1].imshow(m,aspect='auto')
      plt.savefig('foo.png')
