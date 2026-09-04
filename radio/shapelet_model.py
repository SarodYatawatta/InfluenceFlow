import math
import torch

def shapelet_basis(x,beta,n):
        # x: input tensor
        # beta: scale
        # n: order
        # B_n(x,beta) = 1/sqrt(2^n sqrt(pi) n! beta) H_n(x/beta) exp(-0.5 (x/beta)^2)
        Hn=torch.special.hermite_polynomial_h(x/beta,n)
        y=1.0/(math.sqrt(2**n)*math.sqrt(math.pi)*math.factorial(n))*Hn*torch.exp(-0.5*(x/beta)**2)
        return y

def shapelet_bases(uvw, beta, n0):
        # evaluate shapelet basis functions at given uvw coordinates
        # uvw: n_batch x (baselines x 3) : u,v,w
        # beta: scale
        # n0: order
        z=torch.zeros(uvw.shape[0],uvw.shape[1],n0,n0).to(uvw.device)
        for ci in range(n0):
           for cj in range(n0):
              z[:,:,ci,cj]=shapelet_basis(uvw[:,:,0],beta,ci)*shapelet_basis(uvw[:,:,1],beta,cj)

        # return shape: n_batch x baselines x (n0 x n0)
        return z


if 0:
    n_batch=2
    N=100
    B=N*N
    uvw=torch.rand(n_batch,B*3)
    print(uvw.shape)
    uvw1=uvw.view(-1,B,3)
    print(uvw1.shape)
    uu,vv=torch.meshgrid(torch.arange(N)-N/2+0.5,torch.arange(N)-N/2+0.5,indexing='xy')
    uvw1[:,:,0]=uu.flatten()
    uvw1[:,:,1]=vv.flatten()

    beta=10
    M=11
    z=shapelet_bases(uvw1, beta, M)
    print(z.shape)

    Z=z[0,:,10,0].view(N,N)
    import matplotlib.pyplot as plt
    fig=plt.figure(1)
    fig,axs=plt.subplots(1)
    im=axs.imshow(Z,aspect='auto',interpolation='none')
    plt.colorbar(im)
    plt.savefig('im.png')

