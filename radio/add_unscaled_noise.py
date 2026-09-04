#!/usr/bin/env python
import pyrap.tables as pt
import numpy as np
import string


def read_corr(msname,noise_power=1):
  tt=pt.table(msname,readonly=False)
  c=tt.getcol('DATA')
  n=(np.random.normal(-1,1,c.shape)+1j*np.random.normal(-1,1,c.shape))
  # mean should be zero
  n=n-np.mean(n)
  N=np.linalg.norm(n)
  scalefac=np.sqrt(noise_power)/N
  tt.putcol('DATA',c+scalefac*n)
  tt.close()
  

if __name__ == '__main__':
  # addes noise to MS, instead of scaling the noise to get
  # the required noise power, scale the noise to match norm
  #args MS noise_power(noise_power= ||noise||**2)
  import sys
  argc=len(sys.argv)
  if argc==2:
   read_corr(sys.argv[1])
  elif argc==3:
   read_corr(sys.argv[1],float(sys.argv[2]))

  exit()
