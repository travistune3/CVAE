# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.categorical import Categorical
from torch.nn.modules import Module


device = torch.device("cuda")


#%%


class DNN(nn.Module):
    def __init__(self, input_size, layers_data: list,):
        super().__init__()
        self.layers = nn.ModuleList()
        self.input_size = input_size  # Can be useful later ...
        for size, activation, BatchNorm, Dropout in layers_data:
            self.layers.append(nn.Linear(input_size, size, bias=True))
            input_size = size  # For the next layer
            if BatchNorm == True:
                self.layers.append(nn.BatchNorm1d(size, affine=True))
            if activation is not None:
                assert isinstance(activation, Module), \
                    "Each tuples should contain a size (int) and a torch.nn.modules.Module."
                self.layers.append(activation)
            if Dropout is not None:
                assert isinstance(Dropout, Module), \
                    "Each tuples should contain a Droput torch.nn.modules.Module."
                self.layers.append(Dropout)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, input_data):
        # print('start')
        for layer in self.layers:
            input_data = layer(input_data)
        #     print('mean = {:.2f}, std = {:.2f}'.format(input_data.mean(), input_data.std()))
        # print('done')
        return input_data
    

class scale_tanh(nn.Module):
    
    '''
    tanh but scaled to have:
        x=a == max value
        x=b == min value
        x=c == 0
    
    use to constrain final rates to interval 
    
    '''
    def __init__(self, a, b, c):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
                
    def forward(self, x):
        return .5 * ( self.a - self.b) * torch.tanh(x-self.c) + .5*(self.a+self.b)


class SERLU(nn.Module):

    '''
    the activation function we use
    
    https://arxiv.org/pdf/1807.10117.pdf
    
    '''
    def __init__(self):
        super().__init__()
        
        self.lambda_ = 1.0786
        self.alpha_ = 2.90427

        # # returns x if x>0, else 'value'
        self.relu = nn.ReLU()

    def forward(self, x):
        
        return self.lambda_ * (self.relu(x) - self.alpha_ * self.relu(-x) * torch.exp(x))
        
    
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None, dropout = 0):
        super(ResidualBlock, self).__init__()
        
        self.activation = torch.jit.script(SERLU())
        
        self.conv1 = nn.Sequential(
                        nn.Conv1d(in_channels, 
                                  out_channels, 
                                  kernel_size = 17, 
                                  stride = stride, 
                                  padding = 17//2),
                        self.activation)
        self.conv2 = nn.Sequential(
                        nn.Conv1d(out_channels, 
                                  out_channels, 
                                  kernel_size = 17, 
                                  stride = 1, 
                                  padding = 17//2),
        )
        
        self.downsample = downsample
        self.out_channels = out_channels
        
    def forward(self, x):
        
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.activation(out)
        
        return out

class ResNet(nn.Module):
    
    '''
    adapted from https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/
    
    '''
    
    
    def __init__(self, block, layers, dropout_rate = 0):
        super(ResNet, self).__init__()
        
        self.inplanes_0 = 32
        self.inplanes = self.inplanes_0
        
        self.activation = torch.jit.script(SERLU())
        
        self.dropout_rate = 0
        
        self.conv1 = nn.Sequential(
                        nn.Conv1d(1, self.inplanes_0, kernel_size = 33, stride = 8, padding = 16),
                        self.activation)
        
        self.layer0 = self._make_layer(block, self.inplanes_0, layers[0], stride = 2, 
                                       dropout = self.dropout_rate)
        self.layer1 = self._make_layer(block, self.inplanes_0*2, layers[1], stride = 2, 
                                       dropout = self.dropout_rate)
        self.layer2 = self._make_layer(block, self.inplanes_0*4, layers[2], stride = 2, 
                                       dropout = self.dropout_rate)
        self.layer3 = self._make_layer(block, self.inplanes_0*8, layers[3], stride = 2, 
                                       dropout = self.dropout_rate)
                
    def _make_layer(self, block, planes, blocks, stride=1, dropout = 0):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride), 
                # nn.LazyBatchNorm1d()
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dropout = dropout))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        
        x = self.conv1(x)
        
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        out = x.mean(dim=(-1))  # or dim=(2, 3)
        
        return out



class ResidualBlock_DNN(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock_DNN, self).__init__()
        
        self.activation = torch.jit.script(SERLU())
        
        self.lin1 = nn.Sequential(
                        nn.Linear(in_features, out_features, bias = True),
                        self.activation)
        self.lin2 = nn.Sequential(
                        nn.Linear(out_features, out_features, bias = True))
        self.out_features = out_features
        
        
    def forward(self, x):
        
        residual = x
        out = self.lin1(x)
        out = self.lin2(out)
        
        out += residual
        
        out = self.activation(out)
        return out
    

class ResNet_DNN(nn.Module):
    def __init__(self, block, layers, in_size = 1000, out_size = 1000):
        super(ResNet_DNN, self).__init__()
        
        self.w=1024
        
        self.in_size = in_size
        self.out_size = out_size
        
        self.activation = torch.jit.script(SERLU())
        
        self.input = nn.Sequential( nn.Linear(in_size, self.w, bias = True),
                        self.activation)
        
        self.layer0 = self._make_layer(block, layers[0]) 
        self.layer1 = self._make_layer(block, layers[1]) 
        self.layer2 = self._make_layer(block, layers[2]) 
        self.layer3 = self._make_layer(block, layers[3]) 
        
        self.output = nn.Sequential( nn.Linear(self.w, self.out_size, bias = True))
        
    def _make_layer(self, block, blocks):

        layers = []
        layers.append(block(self.w, self.w))
        for i in range(1, blocks):
            layers.append(block(self.w, self.w))
        return nn.Sequential(*layers)
    
    def forward(self, x): 
        
        x = self.input(x) 
        x = self.layer0(x) 
        x = self.layer1(x) 
        x = self.layer2(x) 
        x = self.layer3(x) 
        x = self.output(x)
        return x




class CVAE(nn.Module):
    '''
    https://www.nature.com/articles/s41567-021-01425-7
    
    
    '''
    def __init__(self, signal_length, n_rate_params, latent_size, 
                 N_modes_gmm, scale):
        super(CVAE, self).__init__()
        
        self.latent_dim = latent_size
        self.N_modes_gmm = N_modes_gmm
        self.signal_length = signal_length
        self.n_rate_params = n_rate_params
        
        self.a, self.b, self.c = scale
        self.scale = scale_tanh(self.a, self.b, self.c)
        
        self.tanh = nn.Tanh()
        self.elu = nn.ELU()
        self.softmax = nn.Softmax(dim=1)
        self.flatten = nn.Flatten()
        
        self.eps = 10**(-3)
        
        self.conv_force = ResNet(ResidualBlock, [4, 4, 4, 4], dropout_rate = 0.0) #
        self.conv_output_shape = self.conv_force(torch.rand(2, 1, self.signal_length)).shape[-1]
        
        width = 4096//1 #self.conv_output_shape
        
        self.activation = torch.jit.script(SERLU())
        # self.activation = nn.GELU()
        
        # self.dropout = nn.Dropout(.25)
        # self.dropout = shift_dropout(.00)
        self.dropout = None 
        bn = False
        
        # Q_DNN_layer_list = [(width, self.activation, bn, self.dropout ),
        #                     (width//2, self.activation, bn, self.dropout ),
        #                     (width//4, self.activation, bn, self.dropout ),
        #                     (2 * self.latent_dim, None, False, None ),
        #                   ]
        # self.Q_DNN = DNN(self.conv_output_shape + self.n_rate_params, 
        #                     Q_DNN_layer_list)
        
        self.Q_DNN = ResNet_DNN(ResidualBlock_DNN, [2, 2, 2, 2], 
                                in_size=self.conv_output_shape + self.n_rate_params, 
                                out_size=2 * self.latent_dim)

        
        
        
        # R1_DNN_layer_list = [(width, self.activation, bn, self.dropout ),
        #                       (width//2, self.activation, bn, self.dropout ),
        #                       (width//4, self.activation, bn, self.dropout ), 
        #                       (2*self.latent_dim*self.N_modes_gmm + N_modes_gmm, None, False, None),
        #                   ]
        # self.R1_DNN = DNN(self.conv_output_shape, 
        #                     R1_DNN_layer_list)
        
        
        self.R1_DNN = ResNet_DNN(ResidualBlock_DNN, [2, 2, 2, 2], 
                                in_size=self.conv_output_shape, 
                                out_size=2*self.latent_dim*self.N_modes_gmm+N_modes_gmm)
        
        
        # R2_DNN_layer_list = [(width, self.activation, bn, self.dropout ),
        #                       (width//2, self.activation, bn, self.dropout ),
        #                       (width//4, self.activation, bn, self.dropout ),
        #                       (2*self.n_rate_params*self.N_modes_gmm + N_modes_gmm, None, False, None),
        #                   ]
        # self.R2_DNN = DNN(self.conv_output_shape + self.latent_dim, 
        #                     R2_DNN_layer_list)
        
        
        self.R2_DNN = ResNet_DNN(ResidualBlock_DNN, [2, 2, 2, 2], 
                                in_size=self.conv_output_shape + self.latent_dim, 
                                out_size=2*self.n_rate_params*self.N_modes_gmm+N_modes_gmm)
        
        
        # normal dist with means = 0 and E = I, with dimensionalaity of latent dim, used to regularize latent dim
        self.N_0_I = MultivariateNormal(loc = torch.zeros(self.latent_dim, device=device), 
                                    covariance_matrix=torch.eye(self.latent_dim, device=device))
        
        self.apply(self.weight_init)
        
    def weight_init(self, m):
        
        '''
        the activation functino requires this initillzation
        https://arxiv.org/abs/1607.02488
        
        '''
        
        if isinstance(m, (nn.Conv1d)):
            with torch.no_grad():
                nn.init.kaiming_normal_(m.weight, nonlinearity='selu')
                # m.weight *= .1 
                nn.init.zeros_(m.bias)
            
        if isinstance(m, (nn.Linear)):
            with torch.no_grad():
                nn.init.kaiming_normal_(m.weight, nonlinearity='selu')
                # m.weight *= .1 
                nn.init.zeros_(m.bias)
        return 
              
    def remove_weight_norm (self, m):      
        # print('removed weight norm')
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            pass
            m = torch.nn.utils.remove_weight_norm(m) 
        return
            
    def add_weight_norm (self, m):  
        # print('added weight norm')  
        if isinstance(m, (nn.Linear, nn.Conv1d)):  
             m = torch.nn.utils.weight_norm(m) 
        return  

    def encoder_q(self, x, r): # Q(z|x, c)
        '''
        takes force and rates, does nonlinear dim. reduction
        x: (bs, signal_length)
        c: (bs, n_rate_params)
        '''
        
        x_ = self.conv_force(x)
        r_ = r#self.tanh(self.expand_r(r))
        
        out = self.Q_DNN(torch.cat([x_, r_], 1))
        
        means = out[:,:self.latent_dim]
        logvar = out[:,self.latent_dim:]
        
        return means, logvar

    def encoder_r1(self, x):
        '''
        takes force only, does nonlinear dim. reduction
        x: (bs, signal_length)
        c: (bs, n_rate_params)
        '''
        x_ = self.conv_force(x)
        
        out = self.R1_DNN(x_)
        
        means = out[:,:self.latent_dim*self.N_modes_gmm].reshape(-1, self.N_modes_gmm,self.latent_dim)
        logvar = out[:,self.latent_dim*self.N_modes_gmm:-self.N_modes_gmm].reshape(-1,self.N_modes_gmm,self.latent_dim)
        weights = out[:,-self.N_modes_gmm:]
        
        weights = self.softmax(weights) # transfrom from log_weight to normalized prob
        
        return means, logvar, weights
        
    def decoder_r2(self, x, z): # P(x|z, c)
        '''
        takes force and random nummber from lattent dist, predictts rate
        z: (bs, latent_size)
        c: (bs, singal_length)
        '''
        
        x_ = self.conv_force(x)
        z_ = z 
        
        out = self.R2_DNN(torch.cat([x_, z_], 1))
        
        means = out[:,:self.n_rate_params*self.N_modes_gmm].reshape(-1, self.N_modes_gmm,self.n_rate_params)
        logvar = out[:,self.n_rate_params*self.N_modes_gmm:-self.N_modes_gmm].reshape(-1,self.N_modes_gmm,self.n_rate_params)
        weights = out[:,-self.N_modes_gmm:]
        
        # means = self.scale(means) # constrain to range of real units
        logvar = -self.elu(logvar) - 1 + self.eps
        weights = self.softmax(weights) # transform from log_weight to normalized prob
        
        return means, logvar, weights
    
    
    
    @staticmethod        
    def roll_by_gather(mat,dim, shifts: torch.LongTensor):
        '''
        shift each data point (twitch) in a batch by a random amount
        '''
        # assumes 2D array
        n_rows, n_cols = mat.shape
        
        if dim==0:
            arange1 = torch.arange(n_rows, device=mat.device).view((n_rows, 1)).repeat((1, n_cols))
            arange2 = (arange1 - shifts) % n_rows
            return torch.gather(mat, 0, arange2)
        elif dim==1:
            arange1 = torch.arange(n_cols, device=mat.device).view(( 1,n_cols)).repeat((n_rows,1))
            arange2 = (arange1 - shifts) % n_cols
            return torch.gather(mat, 1, arange2)
    

    def forward(self, x, r):

        # x = self.roll_by_gather(x, 1, 
        #                         torch.randint(0,1000,size = (x.shape[0],1), 
        #                                       device=x.device)
        #                         )  

        # reshape for input to conv network, contigous() means contiguous in memory on gpu for speed 
        x = x.unsqueeze(-1).transpose(1,2).contiguous()
        
        # encoder Q: (x|r) -> z_q
        means_q, logvar_q = self.encoder_q(x, r)
        vars_q = torch.exp(logvar_q) + self.eps
        gmm_z = MultivariateNormal(loc = means_q, 
                                   covariance_matrix=torch.diag_embed(vars_q))
        # get one sample for each data point in batch from latent space
        z_q = gmm_z.rsample()
            
        # encoder r1: (x) -> z_r
        means_r1, logvar_r1, weights_r1 = self.encoder_r1(x)
        vars_r1 = torch.exp(logvar_r1) + self.eps 
        mix = Categorical(probs = weights_r1, validate_args=False)
        comp = MultivariateNormal(loc = means_r1, 
                                  covariance_matrix=torch.diag_embed(vars_r1))
        gmm_r1 = MixtureSameFamily(mix, comp)

        # decoder (x|z_q) -> dist over r, the real rate params space 
        means_r2, logvar_r2, weights_r2 = self.decoder_r2(x, z_q)
        vars_r2 = torch.exp(logvar_r2 ) + self.eps
        mix = Categorical(probs = weights_r2, validate_args=False)
        comp = MultivariateNormal(loc = means_r2, 
                                  covariance_matrix=torch.diag_embed(vars_r2))
        gmm_r2 = MixtureSameFamily(mix, comp)
        
        # we want to minimize kl, kullback liebler divergence aka relative entropy
        # because we want the second encoder to mimic the first
        # K(P,Q) = entropy of P - log prop of Q evaluated at Pi (???)  check
        # print(gmm_r1.log_prob(z_q).min())
        kl = -1 * gmm_z.entropy() - gmm_r1.log_prob(z_q)  
        
        # we also want to maximize the logprob (minimize -logprob) 
        # value of the fitted pdf at the real rate
        # params r
        L = -1. * gmm_r2.log_prob(r)
        # so we minimize kl-L

        # addtional penalty term: kl divergence between Q's latent representation and N(0,1) dist        
        kl_0_I = torch.distributions.kl.kl_divergence(gmm_z, self.N_0_I)
        
        return torch.mean(kl), torch.mean(L), torch.mean(kl_0_I) #+ torch.mean(l2) #, z_q, means_r1, vars_r1, weights_r1, means_r2, vars_r2, weights_r2

    def eval_test_data(self, x):
        
        '''
        
        the difference between this and the forward (training) method is that we use the z output by the r1 network 
        instead of the q network as input to our decoder r2
        
        # '''
        
        x = x.unsqueeze(-1).transpose(1,2).contiguous()
        
        # encoder Q
        pass # don't need ...
        
        # encoder r1: (x) -> z_r
        means_r1, logvar_r1, weights_r1 = self.encoder_r1(x)
        vars_r1 = torch.exp(logvar_r1) + self.eps
        mix = Categorical(probs = weights_r1)
        comp = MultivariateNormal(loc = means_r1, covariance_matrix=torch.diag_embed(vars_r1))
        gmm_r1 = MixtureSameFamily(mix, comp)

        z_r = gmm_r1.sample()

        # decoder (x|z_q) -> dist over r, the real rate params space 
        means_r2, logvar_r2, weights_r2 = self.decoder_r2(x, z_r)
        vars_r2 = torch.exp(logvar_r2) + self.eps

        return  means_r2, vars_r2, weights_r2 


    def eval_train_data(self, x, r):
        
        '''
        
        the difference between this and the forward (training) method is that we use the z output by the r1 network 
        instead of the q network as input to our decoder r2
        
        # '''
        
        x = x.unsqueeze(-1).transpose(1,2).contiguous()
        
        # encoder Q: (x|r) -> z_q
        means_q, logvar_q = self.encoder_q(x, r)
        vars_q = torch.exp(logvar_q) + self.eps
        gmm_z = MultivariateNormal(loc = means_q, covariance_matrix=torch.diag_embed(vars_q))
        z_q = gmm_z.rsample()
        
        # encoder r1: (x) -> z_r
        pass

        # decoder (x|z_q) -> dist over r, the real rate params space 
        means_r2, logvar_r2, logweights_r2 = self.decoder_r2(x, z_q)
        vars_r2 = torch.exp(logvar_r2) + self.eps

        return  means_r2, vars_r2, logweights_r2 



