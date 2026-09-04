import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal,MultivariateNormal

# (try to) use a GPU for computation?
use_cuda=True
if use_cuda and torch.cuda.is_available():
  mydevice=torch.device('cuda')
else:
  mydevice=torch.device('cpu')


def bunch_layers(n_layers, n_dim):
    '''
    helper function to create n_layers, n_dim dimension
    '''
    layers=[]
    for i in range(n_layers):
        layers.append(nn.Linear(n_dim,n_dim))
        layers.append(nn.SiLU())
    return layers

class BaseDist(nn.Module):
    '''
    x ~ p_X(x) base distribution 
    x: Nx1
    '''
    def __init__(self,N,n_metadata,n_hidden=32,depth=3,epsilon=1.0,dropout=True):
        super(BaseDist,self).__init__()
        self.N=N
        # variance = epsilon+sigma
        self.eps=epsilon
        self.depth=depth
        self.dropout=dropout
        self.dropout_rate=0.1
        # limit max value of covariance, to stop blowing up
        self.max_logsigma=1.2
        self.fc1=nn.Linear(N,n_hidden)
        self.fc2=nn.Sequential(*bunch_layers(self.depth,n_hidden))
        if self.dropout:
            self.dp2=nn.Dropout(p=self.dropout_rate)
        self._init_layer(self.fc1)
        self._init_weights(self.fc2)
        # make sure hidden dim of metadata >= n_metadata for the first block
        n_meta_hidden=max(n_hidden,n_metadata)
        self.fcm1=nn.Linear(n_metadata,n_meta_hidden)
        self.fcm2=nn.Sequential(*bunch_layers(self.depth,n_meta_hidden))
        if self.dropout:
            self.dpm2=nn.Dropout(p=self.dropout_rate)
        self._init_layer(self.fcm1)
        self._init_weights(self.fcm2)
        self.fc3=nn.Linear(n_hidden+n_meta_hidden,n_hidden)
        self.fc4=nn.Sequential(*bunch_layers(self.depth,n_hidden))
        if self.dropout:
            self.dp4=nn.Dropout(p=self.dropout_rate)
        self._init_layer(self.fc3)
        self._init_weights(self.fc4)
        self.fcmu=nn.Linear(n_hidden,N)
        # for diagonal covariance (otherwise full covariance)
        self.cov_diag=True
        if self.cov_diag:
           self.fclogsigma=nn.Linear(n_hidden,N)
           self.scalar=torch.tensor(self.eps).to(mydevice)
        else:
           self.fclogsigma=nn.Linear(n_hidden,N*(N+1)//2)
           self.scalar=self.eps*torch.eye(self.N).to(mydevice)
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

    def sample(self,y,metadata, reparameterize=False):
        mu, logsigma=self.forward(y,metadata)
        # to keep sigma going to -> 0, learn delta sigma instead of true sigma
        sigma=logsigma.exp()
        if self.cov_diag:
            mvn=Normal(mu,sigma+self.scalar)
        else:
            batch_size=y.shape[0]
            cov=torch.zeros(batch_size,self.N,self.N).to(mydevice)
            ti=torch.tril_indices(self.N,self.N)
            cov[:,ti[0],ti[1]]=sigma
            mvn=MultivariateNormal(loc=mu,scale_tril=cov+self.scalar)
        if reparameterize:
            probs=mvn.rsample()
        else:
            probs=mvn.sample()

        log_probs=mvn.log_prob(probs)

        return probs, log_probs, mu, logsigma

    def sample_mean(self,y,metadata):
        mu, _=self.forward(y,metadata)

        return mu

    def forward(self,y,z):
        # y:data, z:metadata
        xy=self.fc1(y)
        x=F.silu(xy)
        x=self.fc2(x)
        if self.dropout:
            x=self.dp2(x)
        m=F.silu(self.fcm1(z))
        m=self.fcm2(m)
        if self.dropout:
            m=self.dpm2(m)
        x=F.silu(self.fc3(torch.cat((x,m),1)))
        # add residual connection
        x=self.fc4(x+xy)
        if self.dropout:
            x=self.dp4(x)
        mu=self.fcmu(x)
        logsigma=self.fclogsigma(x)
        logsigma=torch.clamp(logsigma,min=-20,max=self.max_logsigma)

        return mu, logsigma

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

