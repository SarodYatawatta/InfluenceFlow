import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal,MultivariateNormal
import numpy as np
import math

# (try to) use a GPU for computation?
use_cuda=True
if use_cuda and torch.cuda.is_available():
  mydevice=torch.device('cuda')
else:
  mydevice=torch.device('cpu')

class StudentT(nn.Module):
    """
    A Multivariate Student's T distribution with a diagonal covariance matrix.
    The degrees of freedom (nu) is kept constant.
    The mean and diagonal variances remain fully differentiable.
    """
    def __init__(self, dim: int, nu: float = 4.0):
        super().__init__()
        if nu <= 2.0:
            raise ValueError("nu must be > 2 to maintain a well-defined covariance matrix.")
        
        self.dim = dim
        
        # Register nu as a buffer to ensure it moves with the module to CPU/GPU 
        # without tracking optimization gradients.
        self.register_buffer("nu", torch.tensor(float(nu)))
        
        # Precompute static log-gamma normalization factors for speed
        log_gamma_num = math.lgamma((nu + dim) / 2.0)
        log_gamma_den = math.lgamma(nu / 2.0)
        self.log_gamma_constant = log_gamma_num - log_gamma_den

    def log_prob(self, x: torch.Tensor, mean: torch.Tensor, cov_diag: torch.Tensor) -> torch.Tensor:
        """
        Computes the differentiable log-probability density for a batch of samples.
        
        Args:
            x: Input data tensor of shape (batch_size, dim)
            mean: Mean vector parameter of shape (dim,) or (batch_size, dim)
            cov_diag: Covariance diagonal entries of shape (dim,) or (batch_size, dim)
        """
        d = self.dim
        nu = self.nu
        
        # 1. Coordinate-wise Mahalanobis distance computation
        scaled_diffs = ((x - mean) ** 2) / cov_diag
        mahalanobis_dist = torch.sum(scaled_diffs, dim=-1) # Shape: (batch_size,)
        
        # 2. Covariance matrix determinant term: log(det(Sigma)) = sum(log(sigma_i^2))
        log_det_covariance = torch.sum(torch.log(cov_diag), dim=-1)
        
        # 3. Assemble full normalization framework
        log_constant = (self.log_gamma_constant 
                        - 0.5 * d * torch.log(math.pi * nu) 
                        - 0.5 * log_det_covariance)
        
        # 4. Tail density calculation step
        log_main_term = -0.5 * (nu + d) * torch.log(1.0 + mahalanobis_dist / nu)
        
        return log_constant + log_main_term

    def sample(self, num_samples: int, mean: torch.Tensor, cov_diag: torch.Tensor) -> torch.Tensor:
        """
        Generates heavy-tailed sample vectors while preserving the exact 
        mean and diagonal covariance structures.
        """
        device = mean.device
        nu = self.nu
        
        # 1. Variance correction adjustment factor
        variance_correction = (nu - 2.0) / nu
        std_corrected = torch.sqrt(cov_diag * variance_correction) # Shape: (dim,)
        
        # 2. Draw standard spatial uncorrelated Gaussian coordinates
        z_normal = torch.randn(num_samples, self.dim, device=device)
        z_scaled = z_normal * std_corrected
        
        # 3. Generate global scaling multiplier via Gamma sampler: ChiSquared = Gamma(nu/2, 0.5)
        gamma_dist = torch.distributions.Gamma(nu / 2.0, 0.5)
        chi_square_samples = gamma_dist.sample((num_samples, 1)).to(device)
        
        # 4. Compute radial scaling factor
        radial_scale = torch.sqrt(nu / chi_square_samples)
        
        # 5. Broadcast combination with spatial parameters
        return mean + (radial_scale * z_scaled)


def bunch_layers(n_layers, n_dim, batchnorm=False):
    '''
    helper function to create n_layers, n_dim dimension
    '''
    layers=[]
    for i in range(n_layers):
        layers.append(nn.Linear(n_dim,n_dim))
        if batchnorm:
           layers.append(nn.BatchNorm1d(n_dim))
        layers.append(nn.SiLU())
    return layers

class BaseDist(nn.Module):
    '''
    x ~ p_X(x) base distribution 
    x: Nx1, N determined by inputs:
     B=n_stat*(n_stat-1)//2 baselines
     x = data: residual: B*8
     and 
     metadata:
     uvw: B*3
     observed data: B*8
     solutions: n_dir*n_stat*8
     sky: n_dir*3
    '''
    def __init__(self,n_stat=8,n_dir=6,n_hidden=32,depth=6,epsilon=1.0,batchnorm=False,dropout=False):
        '''
        epsilon: add to diagonal of covariance
        batchnorm: if true, enable batchnorm
        dropout: if true, enable dropout (for emulating ensemble)

        '''
        super(BaseDist,self).__init__()
        self.B=n_stat*(n_stat-1)//2
        self.N=self.B*8
        self.n_stat=n_stat
        self.n_dir=n_dir
        self.depth=depth
        self.eps=epsilon
        self.batchnorm=batchnorm
        self.dropout=dropout
        self.dropout_rate=0.1
        # limit max value of covariance, to stop blowing up
        self.max_logsigma=1.2
        # data layers
        self.fc1=nn.Linear(self.N,n_hidden)
        if self.batchnorm:
           self.bn1=nn.BatchNorm1d(n_hidden)
        self.fc2=nn.Sequential(*bunch_layers(self.depth,n_hidden,self.batchnorm))
        if self.dropout:
            self.dp2=nn.Dropout(p=self.dropout_rate)

        self._init_layer(self.fc1)
        self._init_weights(self.fc2)

        # metadata layers (hidden dims vary)
        self.fc_uvw1=nn.Linear(self.B*3,n_hidden)
        if self.batchnorm:
           self.bn_uvw1=nn.BatchNorm1d(n_hidden)
        self.fc_uvw2=nn.Sequential(*bunch_layers(self.depth,n_hidden,self.batchnorm))
        if self.dropout:
            self.dp_uvw2=nn.Dropout(p=self.dropout_rate)

        self._init_layer(self.fc_uvw1)
        self._init_weights(self.fc_uvw2)

        self.fc_obs1=nn.Linear(self.B*8,n_hidden)
        if self.batchnorm:
           self.bn_obs1=nn.BatchNorm1d(n_hidden)
        self.fc_obs2=nn.Sequential(*bunch_layers(self.depth,n_hidden,self.batchnorm))
        if self.dropout:
            self.dp_obs2=nn.Dropout(p=self.dropout_rate)

        self._init_layer(self.fc_obs1)
        self._init_weights(self.fc_obs2)

        self.fc_sol1=nn.Linear(self.n_stat*self.n_dir*8,n_hidden)
        if self.batchnorm:
           self.bn_sol1=nn.BatchNorm1d(n_hidden)
        self.fc_sol2=nn.Sequential(*bunch_layers(self.depth,n_hidden,self.batchnorm))
        if self.dropout:
            self.dp_sol2=nn.Dropout(p=self.dropout_rate)

        self._init_layer(self.fc_sol1)
        self._init_weights(self.fc_sol2)

        n_hidden_sky=self.n_dir*3
        self.fc_sky1=nn.Linear(self.n_dir*3,n_hidden_sky)
        if self.batchnorm:
           self.bn_sky1=nn.BatchNorm1d(n_hidden_sky)
        self.fc_sky2=nn.Sequential(*bunch_layers(self.depth,n_hidden_sky,self.batchnorm))
        if self.dropout:
            self.dp_sky2=nn.Dropout(p=self.dropout_rate)

        self._init_layer(self.fc_sky1)
        self._init_weights(self.fc_sky2)

        self.fc3=nn.Linear(4*n_hidden+n_hidden_sky,2*n_hidden)
        if self.batchnorm:
           self.bn3=nn.BatchNorm1d(2*n_hidden)
        self.fc4=nn.Linear(2*n_hidden,n_hidden)
        if self.batchnorm:
           self.bn4=nn.BatchNorm1d(n_hidden)
        if self.dropout:
            self.dp4=nn.Dropout(p=2.0*self.dropout_rate)
        self.fc5=nn.Sequential(*bunch_layers(self.depth,n_hidden,self.batchnorm))

        self._init_layer(self.fc3)
        self._init_layer(self.fc4)
        self._init_weights(self.fc5)

        self.fcmu=nn.Linear(n_hidden,self.N)
        # Noise model to use 0: Gaussian (diagonal cov) 1: Student's T
        self.noise_model=0
        self.fclogsigma=nn.Linear(n_hidden,self.N)
        self.scalar=torch.tensor(self.eps).to(mydevice)

        # initialize last layers to zero ~ getting N(0,1) dist
        nn.init.xavier_normal_(self.fcmu.weight)
        nn.init.constant_(self.fcmu.bias,0.0)
        nn.init.xavier_normal_(self.fclogsigma.weight)
        nn.init.constant_(self.fclogsigma.bias,0.0)

        self.checkpoint_file='basedist.model'
        self.to(mydevice)

    def _init_layer(self,layer):
        # init a single linear layer
        nn.init.kaiming_normal_(layer.weight,mode='fan_in',nonlinearity='relu')
        if layer.bias is not None:
            nn.init.constant_(layer.bias,0.0)

    def _init_weights(self,network):
        # initialize all layers (do not use this for output layer)
        layers = list(network.children())
        for i, layer in enumerate(layers):
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)


    def forward(self,x,uvw,obs_data,solutions,sky):
        # x:data(=residual), metadata: uvw, obs_data, solutions, sky
        # keep copy of x without activation for residual connection
        if self.batchnorm:
           xres=self.bn1(self.fc1(x))
           x=F.silu(xres)
        else:
           xres=self.fc1(x)
           x=F.silu(xres)
        x=self.fc2(x)
        if self.dropout:
           x=self.dp2(x)

        if self.batchnorm:
           uvw=F.silu(self.bn_uvw1(self.fc_uvw1(uvw)))
        else:
           uvw=F.silu(self.fc_uvw1(uvw))
        uvw=self.fc_uvw2(uvw)
        if self.dropout:
           uvw=self.dp_uvw2(uvw)

        if self.batchnorm:
           obs_data=F.silu(self.bn_obs1(self.fc_obs1(obs_data)))
        else:
           obs_data=F.silu(self.fc_obs1(obs_data))
        obs_data=self.fc_obs2(obs_data)
        if self.dropout:
           obs_data=self.dp_obs2(obs_data)

        if self.batchnorm:
           solutions=F.silu(self.bn_sol1(self.fc_sol1(solutions)))
        else:
           solutions=F.silu(self.fc_sol1(solutions))
        solutions=self.fc_sol2(solutions)
        if self.dropout:
           solutions=self.dp_sol2(solutions)

        if self.batchnorm:
           sky=F.silu(self.bn_sky1(self.fc_sky1(sky)))
        else:
           sky=F.silu(self.fc_sky1(sky))
        sky=self.fc_sky2(sky)
        if self.dropout:
           sky=self.dp_sky2(sky)

        # add residual connection from x:residual
        if self.batchnorm:
           x=F.silu(self.bn3(self.fc3(torch.cat((x,uvw,obs_data,solutions,sky),1))))
           x=F.silu(self.bn4(self.fc4(x)+xres))
        else:
           x=F.silu(self.fc3(torch.cat((x,uvw,obs_data,solutions,sky),1)))
           x=F.silu(self.fc4(x)+xres)
        if self.dropout:
           x=self.dp4(x)
        x=self.fc5(x)
        mu=self.fcmu(x)
        logsigma=self.fclogsigma(x)
        logsigma=torch.clamp(logsigma,min=-20,max=self.max_logsigma)

        return mu, logsigma

    def sample(self,y,uvw,obs_data,solutions,sky, reparameterize=False):
        mu, logsigma=self.forward(y,uvw,obs_data,solutions,sky)
        sigma=logsigma.exp() + self.scalar
        if self.noise_model==0: # Gaussian
            mvn=Normal(mu,sigma)
            if reparameterize:
               samples=mvn.rsample()
            else:
               samples=mvn.sample()

            log_probs=mvn.log_prob(samples)

            return samples, log_probs
        else: # Student's T 
            heavy_dist=StudentT(dim=mu.shape[1],nu=2.1)
            samples=heavy_dist.sample(
                    num_samples=mu.shape[0],
                    mean=mu,
                    cov_diag=sigma
                    )
            if not reparameterize:
                samples=samples.detach()
            log_probs=heavy_dist.log_prob(
                    x=samples,
                    mean=mu,
                    cov_diag=sigma
                    )

            return samples, log_probs

    def sample_mean(self,y,uvw,obs_data,solutions,sky):
        # instead of drawing samples, just use the learned mean
        mu, _ =self.forward(y,uvw,obs_data,solutions,sky)
        return mu


    def save_checkpoint(self,filename=None):
        if filename is None:
           torch.save(self.state_dict(), self.checkpoint_file)
        else:
           torch.save(self.state_dict(), filename)

    def load_checkpoint(self,filename=None):
        if filename is None:
           self.load_state_dict(torch.load(self.checkpoint_file))
        else:
           self.load_state_dict(torch.load(filename))



if 0:
    bd=BaseDist(14,6,n_hidden=100,depth=1,dropout=True)
    print(bd)
    bd.eval()
    for m in bd.modules():
       if isinstance(m,nn.Dropout):
         print(m)
