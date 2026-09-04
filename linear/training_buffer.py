import numpy as np
import pickle

# clip value
CLIP=1000

class TrainingBuffer(object):
    def __init__(self, max_size, n_input=8, n_metadata=8):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.N=n_input
        self.n_meta=n_metadata

        # residual: Nx1
        self.r_= np.zeros((self.mem_size, self.N), dtype=np.float32)
        # log(Eigenvalues): Nx1
        self.log_eig_= np.zeros((self.mem_size, self.N), dtype=np.float32)
        # Influence function: NxN
        self.F_= np.zeros((self.mem_size, self.N, self.N), dtype=np.float32)
        # metadata
        self.metadata_= np.zeros((self.mem_size, self.n_meta), dtype=np.float32)

        # normalized?
        self.norm_fact_= False
        self.r_mu_=0
        self.r_std_=1
        self.metadata_mu_=0
        self.metadata_std_=1


        self.filename='databuffer.npy' # for saving object

    def store_observation(self, r, log_eig, F, metadata):
        index = self.mem_cntr % self.mem_size
        self.r_[index] = r
        self.log_eig_[index] = log_eig
        self.F_[index] = F
        self.metadata_[index] = metadata
        self.mem_cntr += 1

    def calculate_normalization(self):
        if not self.norm_fact_:
           index = self.mem_cntr % self.mem_size
           # catch when buffer is full mem_cntr==mem_size
           if index == 0 and self.mem_cntr > 0:
               index=self.mem_size
           if index > 0:
               self.r_mu_=np.mean(np.clip(self.r_[:index],-CLIP,CLIP),axis=0)
               self.r_std_=np.std(np.clip(self.r_[:index],-CLIP,CLIP),axis=0)
               self.metadata_mu_=np.mean(np.clip(self.metadata_[:index],-CLIP,CLIP),axis=0)
               self.metadata_std_=np.std(np.clip(self.metadata_[:index],-CLIP,CLIP),axis=0)
               self.norm_fact_=True
        return self.r_mu_,self.r_std_,\
                self.metadata_mu_,self.metadata_std_

    def set_normalization(self,r_mu,r_std,metadata_mu,metadata_std):
        self.r_mu_=r_mu
        self.r_std_=r_std
        self.metadata_mu_=metadata_mu
        self.metadata_std_=metadata_std
        self.norm_fact_=True

    def sample_buffer(self, batch_size, normalize=False):
        # return r, log_eig, F as separate numpy arrays
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        r= self.r_[batch]
        log_eig= self.log_eig_[batch]
        F = self.F_[batch]
        metadata = self.metadata_[batch]

        if normalize:
            self.calculate_normalization()
            r=(r-self.r_mu_)/self.r_std_
            metadata=(metadata-self.metadata_mu_)/self.metadata_std_

        return r,log_eig,F,metadata

    def resize(self,new_size):
        if new_size < self.mem_size:
            pass
        else:
           # residual: Nx1
           r=np.zeros((new_size,self.r_.shape[1]), dtype=np.float32)
           r[:self.r_.shape[0],:self.r_.shape[1]]=self.r_
           self.r_=r
           # log(Eigenvalues): Nx1
           log_eig=np.zeros((new_size,self.log_eig_.shape[1]), dtype=np.float32)
           log_eig[:self.log_eig_.shape[0],:self.log_eig_.shape[1]]=self.log_eig_
           self.log_eig_=log_eig
           # metadata n_meta x 1
           metadata=np.zeros((new_size,self.metadata_.shape[1]), dtype=np.float32)
           metadata[:self.metadata_.shape[0],:self.metadata_.shape[1]]=self.metadata_
           self.metadata_=metadata
           # Influence function: NxN
           F=np.zeros((new_size,self.F_.shape[1],self.F_.shape[2]), dtype=np.float32)
           F[:self.F_.shape[0],:self.F_.shape[1],:self.F_.shape[2]]=self.F_
           self.F_=F

           self.mem_size=new_size

    def merge(self,another_filename):
        # merge buffer given by another filename to this
        # assume both buffers are full, mem_cntr=mem_size
        buffer=TrainingBuffer(self.mem_size,self.N,self.n_meta)
        buffer.load_checkpoint(filename=another_filename)

        r1_filled=min(buffer.mem_cntr,buffer.mem_size)
        r_filled=min(self.mem_cntr,self.mem_size)
        self.resize(r_filled+r1_filled)
        self.r_[r_filled:r_filled+r1_filled]=buffer.r_[:r1_filled]
        self.log_eig_[r_filled:r_filled+r1_filled]=buffer.log_eig_[:r1_filled]
        self.metadata_[r_filled:r_filled+r1_filled]=buffer.metadata_[:r1_filled]
        self.F_[r_filled:r_filled+r1_filled]=buffer.F_[:r1_filled]
        self.mem_cntr+=r1_filled

    def save_checkpoint(self,filename=None):
        if filename is None:
           with open(self.filename,'wb') as f:
             pickle.dump(self,f)
        else:
           with open(filename,'wb') as f:
             pickle.dump(self,f)
        
    def load_checkpoint(self,filename=None):
        temp=None
        if filename is None:
           with open(self.filename,'rb') as f:
             temp=pickle.load(f)
        else:
           with open(filename,'rb') as f:
             temp=pickle.load(f)
        if temp is not None:
          self.N=temp.N
          self.n_meta=temp.n_meta
          self.mem_size=temp.mem_size
          self.mem_cntr=temp.mem_cntr
          self.r_=temp.r_
          self.log_eig_=temp.log_eig_
          self.F_=temp.F_
          self.metadata_=temp.metadata_

    def reset(self):
        self.mem_cntr=0
