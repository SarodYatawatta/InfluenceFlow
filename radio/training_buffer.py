import numpy as np
import pickle

# clip value
CLIP=1000

class TrainingBuffer(object):
    # n_stat: stations
    # n_dir: directions
    def __init__(self, max_size, n_stat=8, n_dir=8):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.N=n_stat
        self.K=n_dir
        # baselines
        self.B=self.N*(self.N-1)//2

        # uvw: B*3
        self.uvw_= np.zeros((self.mem_size, self.B*3), dtype=np.float32)
        # unscaled_uvw: B*3
        self.unscaled_uvw_= np.zeros((self.mem_size, self.B*3), dtype=np.float32)
        # observed data: B*8
        self.data_= np.zeros((self.mem_size, self.B*8), dtype=np.float32)
        # solutions : K*N*8
        self.sol_= np.zeros((self.mem_size, self.K*self.N*8), dtype=np.float32)
        # sky: K*3
        self.sky_= np.zeros((self.mem_size, self.K*3), dtype=np.float32)
        # log(Eigenvalues): B*8
        self.log_eig_= np.zeros((self.mem_size, self.B*8), dtype=np.float32)
        # residual data: B*8
        self.res_= np.zeros((self.mem_size, self.B*8), dtype=np.float32)

        # normalized?
        self.norm_fact_=False
        # uvw converted to log() distances?
        self.log_uvw_=False
        self.uvw_mu_=0
        self.uvw_std_=1
        self.data_mu_=0
        self.data_std_=1
        self.sol_mu_=0
        self.sol_std_=1
        self.sky_mu_=0
        self.sky_std_=1
        self.res_mu_=0
        self.res_std_=1

        self.filename='databuffer.npy' # for saving object

    def store_observation(self, uvw, unscaled_uvw, data, solution, sky, log_eig, residual):
        index = self.mem_cntr % self.mem_size
        self.uvw_[index] = uvw
        self.unscaled_uvw_[index] = unscaled_uvw
        self.data_[index] = data
        self.sol_[index] = solution
        self.sky_[index] = sky
        self.log_eig_[index] = log_eig
        self.res_[index] = residual
        self.mem_cntr += 1

    def set_normalization(self,uvw_mu,uvw_std,data_mu,data_std,\
            sol_mu,sol_std,sky_mu,sky_std,res_mu,res_std):
        # set normalization factors to predefined values
        # usefule when evaluating using another buffer
        self.uvw_mu_=uvw_mu
        self.uvw_std_=uvw_std
        self.data_mu_=data_mu
        self.data_std_=data_std
        self.sol_mu_=sol_mu
        self.sol_std_=sol_std
        self.sky_mu_=sky_mu
        self.sky_std_=sky_std
        self.res_mu_=res_mu
        self.res_std_=res_std
        self.norm_fact_=True

    def convert_uvw(self):
        # each uvw (3x1) -> uvw/sqrt(u^2+v^2+w^2) * log(sqrt(u^2+v^2+w^2))
        # to reduce dynamic range
        if not self.log_uvw_:
           filled=self.mem_cntr % self.mem_size
           if filled==0:
               filled=self.mem_size
           for index in range(filled):
              uvw=self.uvw_[index].reshape(self.B,3)
              uvw_amp=np.sqrt(uvw[:,0]**2+uvw[:,1]**2+uvw[:,2]**2)
              uvw[:,0] *=np.log(uvw_amp)/uvw_amp
              uvw[:,1] *=np.log(uvw_amp)/uvw_amp
              uvw[:,2] *=np.log(uvw_amp)/uvw_amp
              self.uvw_[index]=uvw.flatten()
           self.log_uvw_=True

    def calculate_normalization(self):
        if not self.norm_fact_:
           index = self.mem_cntr % self.mem_size
           # catch when buffer is full mem_cntr==mem_size
           if index == 0 and self.mem_cntr > 0:
               index=self.mem_size
           if index > 0:
              # convert all uvw values
              self.convert_uvw()
              # clip all values to reasonable levels before scaling
              uvw=np.clip(self.uvw_[:index],-CLIP,CLIP)
              self.uvw_mu_=np.mean(uvw,axis=0)
              self.uvw_std_=np.std(uvw,axis=0)
              data=np.clip(self.data_[:index],-CLIP,CLIP)
              self.data_mu_=np.mean(data,axis=0)
              self.data_std_=np.std(data,axis=0)
              sol=np.clip(self.sol_[:index],-CLIP,CLIP)
              self.sol_mu_=np.mean(sol,axis=0)
              self.sol_std_=np.std(sol,axis=0)
              sky=np.clip(self.sky_[:index],-CLIP,CLIP)
              self.sky_mu_=np.mean(sky,axis=0)
              self.sky_std_=np.std(sky,axis=0)
              res=np.clip(self.res_[:index],-CLIP,CLIP)
              self.res_mu_=np.mean(res,axis=0)
              self.res_std_=np.std(res,axis=0)
              
              self.norm_fact_=True

        return self.uvw_mu_,self.uvw_std_,\
               self.data_mu_,self.data_std_,\
               self.sol_mu_,self.sol_std_,\
               self.sky_mu_,self.sky_std_,\
               self.res_mu_,self.res_std_

 
    def sample_buffer(self, batch_size, normalize=False, permute=False):
        # return uvw, unscaled_uvw, data, solutions, sky, log_eig, residual as separate numpy arrays
        # if normalize=True, normalize to have about zero mean, unit variance 
        # scale factors determined using the full data, not the batch
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        self.convert_uvw()
        uvw= np.clip(self.uvw_[batch],-CLIP,CLIP)
        unscaled_uvw= self.unscaled_uvw_[batch]
        data= np.clip(self.data_[batch],-CLIP,CLIP)
        solution= np.clip(self.sol_[batch],-CLIP,CLIP)
        sky= np.clip(self.sky_[batch],-CLIP,CLIP)
        # clip log_eig > 0 to 0 (ideally eig = 1 so log_eig=0)
        log_eig= np.clip(self.log_eig_[batch],-CLIP,0)
        residual= np.clip(self.res_[batch],-CLIP,CLIP)

        if normalize:
            self.calculate_normalization()
            uvw=(uvw-self.uvw_mu_)/self.uvw_std_
            data=(data-self.data_mu_)/self.data_std_
            solution=(solution-self.sol_mu_)/self.sol_std_
            sky=(sky-self.sky_mu_)/self.sky_std_
            residual=(residual-self.res_mu_)/self.res_std_

        if permute:
            perm=np.random.permutation(self.B)
            # 3B cols
            idx=np.arange(3*self.B).reshape(self.B,3)
            uvw=uvw[:,idx[perm].flatten()]
            unscaled_uvw=unscaled_uvw[:,idx[perm].flatten()]
            # 8B cols
            idx=np.arange(8*self.B).reshape(self.B,8)
            data=data[:,idx[perm].flatten()]
            residual=residual[:,idx[perm].flatten()]

        return uvw,unscaled_uvw,data,solution,sky,log_eig,residual

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
           self.K=temp.K
           self.B=temp.B
           self.mem_size=temp.mem_size
           self.mem_cntr=temp.mem_cntr
           self.uvw_=temp.uvw_
           self.unscaled_uvw_=temp.unscaled_uvw_
           self.data_=temp.data_
           self.sol_=temp.sol_
           self.sky_=temp.sky_
           self.log_eig_=temp.log_eig_
           self.res_=temp.res_

    def resize(self,new_size):
        assert(new_size > self.mem_size)
        uvw=np.zeros((new_size,self.uvw_.shape[1]), dtype=np.float32)
        uvw[:self.uvw_.shape[0],:self.uvw_.shape[1]]=self.uvw_
        self.uvw_=uvw
        unscaled_uvw=np.zeros((new_size,self.unscaled_uvw_.shape[1]), dtype=np.float32)
        unscaled_uvw[:self.unscaled_uvw_.shape[0],:self.unscaled_uvw_.shape[1]]=self.unscaled_uvw_
        self.unscaled_uvw_=unscaled_uvw
        data=np.zeros((new_size,self.data_.shape[1]), dtype=np.float32)
        data[:self.data_.shape[0],:self.data_.shape[1]]=self.data_
        self.data_=data
        sol=np.zeros((new_size,self.sol_.shape[1]), dtype=np.float32)
        sol[:self.sol_.shape[0],:self.sol_.shape[1]]=self.sol_
        self.sol_=sol
        sky=np.zeros((new_size,self.sky_.shape[1]), dtype=np.float32)
        sky[:self.sky_.shape[0],:self.sky_.shape[1]]=self.sky_
        self.sky_=sky
        log_eig=np.zeros((new_size,self.log_eig_.shape[1]), dtype=np.float32)
        log_eig[:self.log_eig_.shape[0],:self.log_eig_.shape[1]]=self.log_eig_
        self.log_eig_=log_eig
        res=np.zeros((new_size,self.res_.shape[1]), dtype=np.float32)
        res[:self.res_.shape[0],:self.res_.shape[1]]=self.res_
        self.res_=res

        self.mem_size=new_size


    def merge(self,another_filename):
        # merge buffer given by another filename to this
        # assume both buffers are full, mem_cntr=mem_size
        buffer=TrainingBuffer(self.mem_size,self.N,self.K)
        buffer.load_checkpoint(filename=another_filename)

        r1_filled=min(buffer.mem_cntr,buffer.mem_size)
        r_filled=min(self.mem_cntr,self.mem_size)
        self.resize(r_filled+r1_filled)
        self.uvw_[r_filled:r_filled+r1_filled]=buffer.uvw_[:r1_filled]
        self.unscaled_uvw_[r_filled:r_filled+r1_filled]=buffer.unscaled_uvw_[:r1_filled]
        self.data_[r_filled:r_filled+r1_filled]=buffer.data_[:r1_filled]
        self.sol_[r_filled:r_filled+r1_filled]=buffer.sol_[:r1_filled]
        self.sky_[r_filled:r_filled+r1_filled]=buffer.sky_[:r1_filled]
        self.log_eig_[r_filled:r_filled+r1_filled]=buffer.log_eig_[:r1_filled]
        self.res_[r_filled:r_filled+r1_filled]=buffer.res_[:r1_filled]
        self.mem_cntr+=r1_filled

    def clear_zeros(self):
        # remove any entries with zeros (due to incorrect merging)
        valid_indices=(self.uvw_[:,0] != 0)
        print(f'found {np.sum(valid_indices)} valid')
        temp=self.uvw_[valid_indices]
        self.uvw_=temp
        temp=self.unscaled_uvw_[valid_indices]
        self.unscaled_uvw_=temp
        temp=self.data_[valid_indices]
        self.data_=temp
        temp=self.sol_[valid_indices]
        self.sol_=temp
        temp=self.sky_[valid_indices]
        self.sky_=temp
        temp=self.log_eig_[valid_indices]
        self.log_eig_=temp
        temp=self.res_[valid_indices]
        self.res_=temp
        self.mem_size=np.sum(valid_indices)
        self.mem_cntr=self.mem_size
 
    def reset(self):
        self.mem_cntr=0
