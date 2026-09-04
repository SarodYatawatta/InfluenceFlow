import numpy as np
import torch

# AR(1) Structure
def ar1_cov(n, sigma2, rho):
  return sigma2 * rho**abs(np.arange(n)[:, None] - np.arange(n))

def covariance_generator(N):
    # NxN covariance, positive definite
    C=ar1_cov(N, 1.0, 0.5)
    w=np.random.rand(N)+0.5
    for ci in range(N):
        C[ci] *=w[ci]
    return torch.FloatTensor(C)


if 0:
   import matplotlib.pyplot as plt
   fig=plt.figure(1)
   fig,axs=plt.subplots(1)
   N=15
   C=covariance_generator(N)
   print(torch.linalg.cholesky(C))
   axs.imshow(C.detach().cpu().numpy(),aspect='auto')
   plt.savefig('foo.png')
