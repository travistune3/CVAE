# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 08:24:36 2023

@author: ttune3
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 13:16:09 2023

@author: ttune3
"""


from scipy import stats
from joblib import load

import torch
import torch.utils.data
from torch import nn, optim

import numpy as np
from torch.nn.modules import Module
import scipy

import sklearn


import pdb

import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns

from matplotlib.lines import Line2D

from torch.distributions.categorical import Categorical

from torch.distributions.multivariate_normal import MultivariateNormal

from sklearn.cluster import MeanShift, estimate_bandwidth

from torch.distributions.mixture_same_family import MixtureSameFamily
import time

# import fuckit

np.set_printoptions(suppress=True)

from math import nan

torch.backends.cudnn.benchmark = True

# cuda setup
device = torch.device("cuda")
kwargs = {'num_workers': 0, 'pin_memory': False} 

import sys
# sys.path.append('D:\OneDrive - UW\Daniel_Group\cvae_test')
from CVAE import CVAE

from sklearn.base import BaseEstimator, TransformerMixin
    
    # define the transformer
class Std_scaler(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        print('Initialising transformer...')
        
    def fit(self, X, y = None):
        self.mean = X.mean()
        self.std = X.std()
        return self
    
    def transform(self, X):
        print ('transform')
        
        return (X - self.mean)/self.std
    
    def inverse_transform(self, X):
        print ('inv_transform')
        
        return self.std * X + self.mean
    
    

#%% load training and testing data

# data available at dryad DOI: 10.5061/dryad.d51c5b0bj

# full dataset all rates 1 million points
training_dataset = torch.load(r'E:\datasets\expanded_3\training_datasetMyGlobalScaler_feb29_2024_paper.pt')
testing_dataset = torch.load(r'E:\datasets\expanded_3\testing_datasetMyGlobalScaler_feb29_2024_paper.pt')
# use sclaer.inverse_transform on stress (training_dataset.tensors[0]) to get back original stress in mN/mm2
# use scaler.transform to process new data for cvae (in units of mN/mm2), saved testing dataset already transformed 
scaler = load(r'E:\datasets\expanded_3\std_scalerMyGlobalScaler_feb29_2024_paper.bin')


print(training_dataset.tensors[0].shape) # [data_points, time_len]
print(training_dataset.tensors[1].shape) # [data_points, targets]

print(testing_dataset.tensors[0].shape) # [data_points, time_len]
print(testing_dataset.tensors[1].shape) # [data_points, targerts]





#%%

batch_size = 2048

train_loader = torch.utils.data.DataLoader(
    training_dataset,
    batch_size=batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    testing_dataset,
    batch_size=batch_size, shuffle=True, **kwargs)


i = np.random.randint(0,training_dataset.tensors[0].shape[0]+1)

plt.plot(training_dataset.tensors[0][i,:].cpu().numpy(), label = 'transformed for cvae (z scored)')
plt.plot(scaler.inverse_transform(training_dataset.tensors[0][i,:].cpu().numpy().reshape(1,-1)).T, label = 'original (mN/mm2)')
plt.legend()

#%%


def LSUV_(model, test_loader, apply_only_to=['Conv', 'Linear', 'Bilinear'],
          std_tol=0.1, max_iters=10, do_ortho_init=False, logging_FN=print):
    r"""
    
    https://github.com/glassroom/torch_lsuv_init
    
    Applies layer sequential unit variance (LSUV), as described in
    `All you need is a good init` - Mishkin, D. et al (2015):
    https://arxiv.org/abs/1511.06422

    Args:
        model: `torch.nn.Module` object on which to apply LSUV.
        data: sample input data drawn from training dataset.
        apply_only_to: list of strings indicating target children
            modules. For example, ['Conv'] results in LSUV applied
            to children of type containing the substring 'Conv'.
        std_tol: positive number < 1.0, below which differences between
            actual and unit standard deviation are acceptable.
        max_iters: number of times to try scaling standard deviation
            of each children module's output activations.
        do_ortho_init: boolean indicating whether to apply orthogonal
            init to parameters of dim >= 2 (zero init if dim < 2).
        logging_FN: function for outputting progress information.

    Example:
        >>> model = nn.Sequential(nn.Linear(8, 2), nn.Softmax(dim=1))
        >>> data = torch.randn(100, 8)
        >>> LSUV_(model, data)
    """

    matched_modules = [m for m in model.modules() if any(substr in str(type(m)) for substr in apply_only_to)]

    if do_ortho_init:
        logging_FN(f"Applying orthogonal init (zero init if dim < 2) to params in {len(matched_modules)} module(s).")
        for m in matched_modules:
            for p in m.parameters():                
                torch.nn.init.orthogonal_(p) if (p.dim() >= 2) else torch.nn.init.zeros_(p)

    logging_FN(f"Applying LSUV to {len(matched_modules)} module(s) (up to {max_iters} iters per module):")

    def _compute_and_store_LSUV_stats(m, inp, out):
        m._LSUV_stats = { 'mean': out.detach().mean(), 'std': out.detach().std() }

    was_training = model.training
    model.train()  # sets all modules to training behavior
    with torch.no_grad():
        for i, m in enumerate(matched_modules):
            with m.register_forward_hook(_compute_and_store_LSUV_stats):
                for t in range(max_iters):
                    data = next(iter(test_loader))
                    
                    _ = model(data[0], data[1])  # run data through model to get stats
                    mean, std = m._LSUV_stats['mean'], m._LSUV_stats['std']
                    if abs(std - 1) < std_tol:
                        break
                    
                    m.weight.data /= std + 10**-2
                    
                    
            # logging_FN(f"Module {i:2} after {(t+1):2} itr(s) | Mean:{mean:7.3f} | Std:{std:6.3f} | {type(m)}")
            delattr(m, '_LSUV_stats')

    if not was_training: model.eval()


#%% define training function 


def alpha_annealing(epoch, ramp_period = 10, constant_period = 0): 
    
    if epoch < 0: 
        return .1 
    
    if epoch >= ramp_period: 
        return 1. 
    
    time = epoch % (ramp_period + constant_period) 
    
    if 0 <= time < ramp_period: 
        alpha = 1./(ramp_period - 0) * (time-0) 
    
    if ramp_period <= time < (ramp_period + constant_period): 
        alpha = 1. 
        
    if alpha <= 0:
        alpha = .1 
        
    return alpha
     
def train_both(epoch, alpha):
    
    
    model.train()
    train_loss = 0
    KL_total = 0
    L_total = 0
    KL2_total = 0
    L2_norm = 0
    
    lambda_ = .000*alpha
    
    start = time.time()
    
    scaler = torch.cuda.amp.GradScaler()
    for batch_idx, (data, labels) in enumerate(train_loader):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pass
        with torch.autocast(device_type='cuda', dtype=torch.float16): 
            data, labels = data.to(device), labels.to(device)
            KL, L, KL_2 = model(data, labels) 
            KL_total += KL.item() 
            L_total += L.item() 
            KL2_total += KL_2.item() 
            L2_norm += np.mean([p.pow(2.0).sum().item() for p in model.parameters()])
            
            loss = L + alpha * KL + lambda_ * KL_2
            
        train_loss += loss.item() 
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()

    # average over number of batches 
    L_total = L_total/(batch_idx+1)
    KL_total = KL_total/(batch_idx+1)
    KL2_total = KL2_total/(batch_idx+1)
    L2_norm = L2_norm/(batch_idx+1)
    
    # calculate val error with 
    Val_KL_Total = 0
    Val_L_Total = 0  
    Val_KL2_Total = 0   
    model.eval() 
    with torch.no_grad(): 
        for batch_idx, (data, labels) in enumerate(test_loader):
            pass
            # if batch_idx>20: 
            #     break
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                data, labels = data.to(device), labels.to(device)
                Val_KL, Val_L, Val_KL_2 = model(data, labels) 
                Val_KL_Total += Val_KL.item()
                Val_L_Total += Val_L.item()
                Val_KL2_Total += Val_KL_2.item()
            # break
    
    # average over number of batches 
    Val_L_Total = Val_L_Total/(batch_idx+1)
    Val_KL_Total = Val_KL_Total/(batch_idx+1)
    Val_KL2_Total = Val_KL2_Total/(batch_idx+1)
                
    end = time.time()
    
    print('====> Epoch {}: Log Prob: {:.2f} ({:.2f}), KL: {:.2f} ({:.2f}), Sum: {:.2f} ({:.2f}), Lrn. Gap: {:.2f}, KL2: {:.2f} ({:.2f}), Model L2 Norm: {:.2f}, time: {:.0f}'.format(
          epoch, L_total, Val_L_Total, KL_total, Val_KL_Total, (L_total + KL_total), (Val_L_Total + Val_KL_Total),
          (Val_L_Total + Val_KL_Total) - (L_total + KL_total),
          KL2_total, Val_KL2_Total, L2_norm, end-start ))
        
    model.train()
    
    return  loss, L_total, Val_L_Total, KL_total, Val_KL_Total, KL2_total, Val_KL2_Total, L2_norm 

# train_loss/(batch_idx+1) #L_total/batch_idx, KL_total/batch_idx

print('trainign loop')

#%% define model

model = CVAE(signal_length = training_dataset.tensors[0].shape[1], 
             n_rate_params = training_dataset.tensors[1].shape[1], 
             latent_size = 15, 
             N_modes_gmm = 32, 
             scale = (training_dataset.tensors[1].max()*1.1, 
                      training_dataset.tensors[1].min()*1.1, 
                      training_dataset.tensors[1].mean()) 
             ).to(device) 

print(model.conv_output_shape)



# model.apply(model.remove_weight_norm)
# want to inititlaize weights randomly, but sill have each layer output mean=0, std=1
# LSUV_(model, test_loader, max_iters=10, std_tol=0.1)


# # partition each wieght matrix into a direction and magnitidue during training 
# remove wieght norm when saving 
try: 
    model.apply(model.add_weight_norm)
    print('weight norm added')
except RuntimeError:
    print('weight norm already added')


# only weight decay magnitute parameters (?), use this with weight norm
decay = list()
no_decay = list()
for name, param in model.named_parameters():
    # print('checking {}'.format(name))
    if hasattr(param,'requires_grad') and not param.requires_grad:
        continue
    if 'weight_g' in name:
        decay.append(param)
    else:
        no_decay.append(param)
# return decay, no_decay


parameters = [{'params': no_decay, 'weight_decay': 0},
              {'params': decay, 'weight_decay': .010}]

optimizer = optim.AdamW(parameters,
                        lr = 1e-6, 
                        amsgrad=False, 
                        eps = 1e-4) 


PATH = r'E:\trained_models\june_3_2024\best_model_best.pth'


L2_Loss = []
L_Loss = [] 
KL_Loss = [] 
Val_L_Loss = [] 
Val_KL_Loss = [] 
KL2_Loss = [] 
Val_KL2_Loss = [] 
L2_norm = []


print('initizle model')

#%% load previous saved model

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
L_Loss = checkpoint['L_Loss']
KL_Loss = checkpoint['KL_Loss']
Val_L_Loss = checkpoint['Val_L_Loss']
Val_KL_Loss = checkpoint['Val_KL_Loss']
KL2_Loss = checkpoint['KL2_Loss']
Val_KL2_Loss = checkpoint['Val_KL2_Loss']
L2_norm = checkpoint['L2_norm']

model.train()

print('load saved model')
    



#%%

# # partition each wieght matrix into a direction and magnitidue during training 
# remove wieght norm when saving 
# try: 
#     model.apply(model.add_weight_norm)
# except RuntimeError:
#     print('weight norm already added')

# log prob loss
# best_loss = 10**6 

# max_weight_decay = 10 
# min_weight_decay = 10**-4 

# ♠epoch = -50 
p0 = 4000 
p = p0 
i_ = 0 

while epoch < 25000: 
    
    # if best_loss > 0:
    #     p=False
    #     epoch = -50
    # else:
    #     pass
    
    
    
    # if epoch > 125:
                
        # for g in optimizer.param_groups: 
        #     pass
        #     g['lr'] = 1e-5
    
    
    alpha = alpha_annealing(epoch, ramp_period = 50) 
    
    loss, L_total, Val_L_Total, KL_total, Val_KL_Total, KL2_total, Val_KL2_Total, L2_mean = train_both(epoch, alpha) 
    
    L_Loss.extend([L_total]) 
    KL_Loss.extend([KL_total]) 
    Val_L_Loss.extend([Val_L_Total]) 
    Val_KL_Loss.extend([Val_KL_Total]) 
    KL2_Loss.extend([KL2_total]) 
    Val_KL2_Loss.extend([Val_KL2_Total]) 
    L2_norm.extend([L2_mean]) 
    
    epoch += 1 
    
    best_loss = L_total
    
    # delta_error = np.subtract(np.add(Val_L_Loss, Val_KL_Loss), np.add(L_Loss, KL_Loss))
    # weight_decay = np.mean(delta_error[-1:])
    # weight_decay = max(min(weight_decay, max_weight_decay), min_weight_decay)
    # for g in optimizer.param_groups:
    #     pass
    #     g['weight_decay'] = weight_decay
    
    if i_>=50: 
        i_= 0    
        print(r'starting saving, dont quit') 
        # remove weight norm to save
        # model.apply(model.remove_weight_norm)
        torch.save({ 
            'epoch': epoch, 
            'model_state_dict': model.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict(), 
            'L_Loss': L_Loss, 
            'KL_Loss': KL_Loss, 
            'Val_L_Loss': Val_L_Loss, 
            'Val_KL_Loss': Val_KL_Loss, 
            'KL2_Loss': KL2_Loss, 
            'Val_KL2_Loss': Val_KL2_Loss, 
            'L2_norm': L2_norm, 
            }, PATH) 
        # add back weight norm to keep traiing 
        # model.apply(model.add_weight_norm)
        print('saved') 
        
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        i = 0
        ax1.plot(L_Loss[i:], c='C0', label = 'L')
        ax1.plot(Val_L_Loss[i:], ':', c='C0', label = 'Val L')
        ax1.plot(KL_Loss[i:], c='C2', label = 'KL')
        ax1.plot(Val_KL_Loss[i:], ':', c='C2', label = 'Val KL')
        ax1.plot(np.add(L_Loss[i:], KL_Loss[i:]), c='C3', label = 'L+KL')
        ax1.plot(np.add(Val_L_Loss[i:], Val_KL_Loss[i:]), ':', c='C3', label = 'val L+KL')
        # ax1.plot(KL2_Loss[i:], c='C4', label = 'KL 2')
        # ax1.plot(Val_KL2_Loss[i:], ':', c='C4', label = 'Val KL 2')
        ax2.plot(L2_norm[i:], c='C5', label = 'Model L2 Norm')
        ax2.set_ylim([0,1.1*max(L2_norm)])
        ax1.set_ylim([-20,35])
        fig.legend(fontsize = 'xx-small')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Error Terms')
        ax2.set_ylabel('L2 Norm', color='C5')
        plt.show()    
        
        
    i_ = i_ + 1
#%% plot posterior NEW WAY

model.eval()
test_loss = 0
plt.figure()
i=0
with torch.no_grad():
    while i < 1:
        # print(data)
        data, labels = next(iter(test_loader))
        
        plt.plot(data[0:1].cpu().detach().numpy().flatten())
        i+=1
        plt.plot((scaler.var_**.5)*data[0:1].cpu().detach().numpy().flatten()+scaler.mean_)
# plt.ylim([-1,100])
#     plt.ylim([stress.min(),stress.max()])
plt.show()





#%% plot posterior NEW WAY 




model.eval()
n_samples = 2500



for i in [0]:
    data_ = data[i:i+1]
    labels_ = labels[i:i+1]
    # means_r2, vars_r2, weights_r2 = model.eval_train_data(data_, labels_)

    fig, ax = plt.subplots()
    ax.plot((scaler.var_**.5)*data_.cpu().numpy().flatten() + scaler.mean_)
    
    # ax.xaxis.set_ticklabels([])
    # ax.yaxis.set_ticklabels([])
    
    plt.ylim([-.5, 120])
    
    # plt.ylim([(scaler.var_**.5*training_dataset.tensors[0].cpu().numpy() + scaler.mean_).min() - .2, 
              # (scaler.var_**.5*training_dataset.tensors[0].cpu().numpy() + scaler.mean_).max() + .5])
    plt.savefig(r'E:\ttune3\OneDrive - UW\Daniel_Group\cvae_test\stress.svg')
    # single = MultivariateNormal(loc = means_r2[:,-1,:], covariance_matrix=torch.diag_embed(vars_r2[:,-1,:]))
    # points_torch = np.zeros([n_samples,labels_.shape[1]])
    points_torch = [] 
    k = 0
    while len(points_torch) < n_samples:
        means_r2, vars_r2, weights_r2 = model.eval_test_data(data_)
        mix = Categorical(probs = weights_r2)
        comp = MultivariateNormal(loc = means_r2, covariance_matrix=torch.diag_embed(vars_r2))
        gmm_r2 = MixtureSameFamily(mix, comp)
        
        # single = MultivariateNormal(loc = means_r2[:,0,:], 
        #                             covariance_matrix=torch.diag_embed(vars_r2[:,0,:]))
        
        # points = single.sample(torch.tensor([1000])).cpu().numpy().squeeze(-2)
        
        single_point = gmm_r2.sample().cpu().numpy() 
        if np.all(single_point < 2) and np.all(single_point > -1):
        # points = gmm_r2.sample(torch.tensor([1000])).cpu().numpy().squeeze(-2)
            points_torch.extend(single_point)
        k += 1
        
        if k>2*n_samples:
            print(k)
            break
    print('points done')
    points_torch = np.array(points_torch)    
    points = pd.DataFrame(points_torch)
    
    points[points < training_dataset.tensors[1].min().cpu().numpy()] = nan
    points[points > training_dataset.tensors[1].max().cpu().numpy()] = nan
    points = points.dropna()

    # points = points.head(8000)
    if points.shape[0]<8000:
        print(r'Warning: ' + str(points.shape[0]) + r'<8000 samples')

    points = points
    g = sns.PairGrid(points, corner=True)
    color_for_trainingset = 'blue'
    # color_for_trainingset = sns.color_palette('husl', 2) [-1] # this is the color from the question
    # g.map_upper(sns.scatterplot, alpha=0.2, color=color_for_trainingset)
    g.map_lower(sns.kdeplot, color=color_for_trainingset)
    g.map_diag(sns.kdeplot, lw=3, color=color_for_trainingset)
    
    kde = scipy.stats.gaussian_kde(points.T)
    #estimate peak of distribution
    bandwidth = estimate_bandwidth(points, quantile=0.2, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(points)
    cluster_centers = ms.cluster_centers_
    
    g.data = pd.DataFrame(labels_.cpu().numpy()).iloc[:1]
    # g.data = data[data['type'] == 'Target 1']
    # g.map_upper(sns.scatterplot, alpha=1, color='red')
    g.map_lower(sns.scatterplot, alpha=1, color='red', zorder=30, s=500)
    
    handles = [Line2D([], [], color='red', ls='', marker='x', label='target'),
               Line2D([], [], color=color_for_trainingset, lw=3, label='training set')]
    g.add_legend(handles=handles)
    
    for ax in g.axes.flat:
        try:
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
            ax.set_xlim((training_dataset.tensors[1].min().cpu().numpy(),training_dataset.tensors[1].max().cpu().numpy()))
        except:
            pass
        try:
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
            ax.set_ylim((training_dataset.tensors[1].min().cpu().numpy(), training_dataset.tensors[1].max().cpu().numpy()))
        except:
            pass
    
    
    plt.savefig(r'E:\ttune3\OneDrive - UW\Daniel_Group\cvae_test\test.svg')
    
    plt.show()



#%% prob-prob plot


n_points = 1000

model.eval() 
with torch.no_grad(): 
    for batch_idx, (data, labels) in enumerate(test_loader):
        pass
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            
            data, labels = data.to(device), labels.to(device) 
            
            points = None
            
            for i in range(n_points):
                means_r2, vars_r2, weights_r2 = model.eval_test_data(data)
                mix = Categorical(probs = weights_r2)
                comp = MultivariateNormal(loc = means_r2, covariance_matrix=torch.diag_embed(vars_r2))
                gmm_r2 = MixtureSameFamily(mix, comp)
                
                single_point =  gmm_r2.sample().unsqueeze(-1)
                
                if points is None:
                    points = single_point
                else:
                    points = torch.cat([points, single_point], -1)
        break
    
# plt.hist(points[0,0,:].detach().cpu().numpy(), bins=100)
x_ = np.linspace(0, 1, data.shape[0])
plt.figure()
for i in range(labels.shape[1]):
    za = (points[:,i,:] < labels[:,i].unsqueeze(-1)).sum(1)/n_points
    za = torch.sort(za)[0]
    plt.plot(x_,za.detach().cpu().numpy(), label = i)
    plt.legend()
plt.plot(x_,x_, '--k')
plt.ylim([0,1])
plt.xlim([0,1])
# plt.savefig(r'E:\ttune3\OneDrive - UW\Daniel_Group\cvae_test\prob_prob.svg')

plt.show()



# x_ = np.linspace(0, 1, data.shape[0])
# plt.figure()
# for i in range(9):
    
    
#     za = (torch.sort(points[:,i,:])[0] < labels[:,i].unsqueeze(-1)).sum(1)/n_points
#     za = torch.sort(za)[0]
#     plt.plot(x_,za.detach().cpu().numpy(), label = i)
#     plt.legend()
# plt.plot(x_,x_, '--k')


#%% plot latent 

model.eval()

means_q, logvar_q = model.encoder_q(x, r)
vars_q = torch.exp(logvar_q) + model.eps
gmm_z = MultivariateNormal(loc = means_q, covariance_matrix=torch.diag_embed(vars_q))
z_q = gmm_z.rsample()
    
    
# encoder r1: (x) -> z_r
means_r1, logvar_r1, weights_r1 = model.encoder_r1(x)
vars_r1 = torch.exp(logvar_r1) + model.eps
mix = Categorical(probs = weights_r1)
comp = MultivariateNormal(loc = means_r1, covariance_matrix=torch.diag_embed(vars_r1))
gmm_r1 = MixtureSameFamily(mix, comp)

print(-1. * gmm_z.entropy() - gmm_r1.log_prob(z_q))

points_q = gmm_z.sample(torch.tensor([500])).cpu().numpy().squeeze(-2)
points_q = pd.DataFrame(points_q)

points_r1 = gmm_r1.sample(torch.tensor([500])).cpu().numpy().squeeze(-2)
points_r1 = pd.DataFrame(points_r1)


min_ = 5  # min(points_q.min().min(), points_r1.min().min()) * 1.05 
max_ = -min_  # min(points_q.max().max(), points_r1.max().max()) * 1.05 

g = sns.PairGrid(points_q, corner=True)
color_for_trainingset = 'blue'
# color_for_trainingset = sns.color_palette('husl', 2) [-1] # this is the color from the question
# g.map_upper(sns.scatterplot, alpha=0.2, color=color_for_trainingset)
g.map_lower(sns.kdeplot, color=color_for_trainingset)
g.map_diag(sns.kdeplot, lw=3, color=color_for_trainingset)

for ax in g.axes.flat:
    try:
        ax.set_xlim((min_, max_))
    except:
        pass
    try:
        ax.set_ylim((min_, max_))
    except:
        pass

g = sns.PairGrid(points_r1, corner=True)
color_for_trainingset = 'red' 
# color_for_trainingset = sns.color_palette('husl', 2) [-1] # this is the color from the question
# g.map_upper(sns.scatterplot, alpha=0.2, color=color_for_trainingset)
g.map_lower(sns.kdeplot, color=color_for_trainingset)
g.map_diag(sns.kdeplot, lw=3, color=color_for_trainingset)

# min_ = -5 #min(points_q.min().min(), points_r1.min().min())*1.05
# max_ =  5 #min(points_q.max().max(), points_r1.max().max())*1.05
for ax in g.axes.flat:
    try:
        ax.set_xlim((min_, max_))
    except:
        pass
    try:
        ax.set_ylim((min_, max_))
    except:
        pass

model.train()




#%% inference of experiemental twitches 

# mean_I61q
# twitches = scaler.transform(np.roll(mean_I61q, -100,1)); type_ = r'I61Q'
# twitches = scaler.transform(np.roll(ave_wild_types,-100,1)); type_ = r'wild_type'
# twitches = np.vstack((scaler.transform(np.roll(mean_I61q, -100,1)).mean(0).flatten(),scaler.transform(np.roll(ave_wild_types,-100,1)).mean(0).flatten()))

from scipy.signal import butter, sosfiltfilt

def butter_bandpass(highcut, fs, order=5):
        nyq = 0.5 * fs
        # low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [high], analog=False, btype='low', output='sos')
        # sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

def butter_bandpass_filter(data, highcut, fs, order=5):
        sos = butter_bandpass(highcut, fs, order=order)
        y = sosfiltfilt(sos, data, axis=0)
        return y
    
    


stress = train_loader.dataset.tensors[0].detach().cpu().numpy()
rates = train_loader.dataset.tensors[1].detach().cpu().numpy()

n = stress.shape[1]

shift = -100

WT_exp_stress = scaler.transform(np.roll(ave_wild_types, shift,1)[:,:n]).mean(0).flatten()
# WT_exp_stress_filtered = butter_bandpass_filter(scaler.transform(np.roll(ave_wild_types, shift,1)[:,:n]).mean(0).flatten().T, 8, 1000, 2)
WT_exp_stress_filtered = scaler.transform(butter_bandpass_filter(np.roll(ave_wild_types, shift,1)[:,:n].mean(0).flatten().T, 
                                                  24, 1000, 4).reshape(1,-1)).flatten()
WT_best_sim = stress[np.argmin(np.linalg.norm(stress - WT_exp_stress, axis=1))]
WT_best_label = rates[np.argmin(np.linalg.norm(stress - WT_exp_stress, axis=1))]

I61Q_stress = scaler.transform(np.roll(mean_I61q, shift,1)[:,:n]).mean(0).flatten()
I61Q_exp_stress_filtered = butter_bandpass_filter(scaler.transform(np.roll(mean_I61q, shift,1)[:,:n]).mean(0).flatten().T, 
                                                  24, 1000, 4)
I61Q_best_sim = stress[np.argmin(np.linalg.norm(stress - I61Q_stress, axis=1))]
I61Q_best_label = rates[np.argmin(np.linalg.norm(stress - I61Q_stress, axis=1))]





names = [r'WT_exp', 
         r'WT_filt', 
         r'WT_sim',
         r'I61Q_exp', 
         r'I61Q_filt', 
         r'I61Q_sim',]

#%%


plt.plot(WT_exp_stress, 'b-', label = names[0])
plt.plot(WT_exp_stress_filtered, 'b:', label = names[1])
plt.plot(WT_best_sim, 'b--', label = names[2])
# plt.plot(I61Q_stress, 'r-', label = names[3])
# plt.plot(I61Q_exp_stress_filtered, 'r:', label = names[4])
# plt.plot(I61Q_best_sim, 'r--', label = names[5])
plt.legend()
plt.show()

# WT_exp_stress_filtered = WT_best_sim
# WT_best_sim = butter_bandpass_filter(WT_best_sim, 12, 1000, 2)

plt.plot(scaler.inverse_transform(WT_exp_stress.reshape(1, -1)).T, 'b-', label = names[0])
plt.plot(scaler.inverse_transform(WT_exp_stress_filtered.reshape(1, -1)).T, 'b:', label = names[1])
plt.plot(scaler.inverse_transform(WT_best_sim.reshape(1, -1)).T, 'b--', label = names[2])
plt.plot(scaler.inverse_transform(I61Q_stress.reshape(1, -1)).T, 'r-', label = names[3])
plt.plot(scaler.inverse_transform(I61Q_exp_stress_filtered.reshape(1, -1)).T, 'r:', label = names[4])
plt.plot(scaler.inverse_transform(I61Q_best_sim.reshape(1, -1)).T, 'r--', label = names[5])
plt.legend()

# name = r'E:\ttune3\OneDrive - UW\Daniel_Group\cvae_test\l2_best_training_set.svg'
# plt.savefig(name)

twitches = [WT_exp_stress,
            WT_exp_stress_filtered,
            WT_best_sim,
            I61Q_stress,
            I61Q_exp_stress_filtered,
            I61Q_best_sim]

# plt.plot(scaler.inverse_transform(twitches).T)
# plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', ])
# # plt.ylim([-1,4])
# ax = plt.gca()
# ax.axes.xaxis.set_ticklabels([])
# ax.axes.yaxis.set_ticklabels([])
# name = r'E:\ttune3\OneDrive - UW\Daniel_Group\cvae_test\exp_mean_twitches_cvae.svg'
# plt.savefig(name)
# plt.show()
# data = scaler.transform(ttt[:,None].T).flatten()



#%%

model.eval()

n_samples = 500



list_of_dataframes = []
cluster_centers_df = []
r = [0,3]
for i, data in enumerate(twitches):
    if i in r:
        print("===============================================")    
        # print(i) 
        pass
        data_ = torch.tensor(data.copy(), dtype=torch.float32).to(device).unsqueeze(-1).T
        # data_ = torch.tensor(ave_wild_types_transformed.mean(0), dtype=torch.float32).to(device).unsqueeze(-1).T
        # labels_ = labels[i:i+1]
        # means_r2, vars_r2, weights_r2 = model.eval_train_data(data_, labels_)
    
        fig, ax = plt.subplots()
        
        ax.plot(scaler.inverse_transform(data_.cpu().numpy().flatten().reshape(1, -1)).T, 'b-', label = names[0])
        # ax.plot((scaler.std**.5)*data_.cpu().numpy().flatten() + scaler.mean, label = names[i])
        # ax.plot((scaler.var_**.5)*data_.cpu().numpy().flatten() + scaler.mean_, label = names[i])
        plt.ylim([-5,60])
        plt.legend()
        plt.show()
        # ax.xaxis.set_ticklabels([])
        # ax.yaxis.set_ticklabels([])
        # plt.ylim([-.5, 120])
        
        # plt.ylim([(scaler.var_**.5*training_dataset.tensors[0].cpu().numpy() + scaler.mean_).min() - .2, 
                  # (scaler.var_**.5*training_dataset.tensors[0].cpu().numpy() + scaler.mean_).max() + .5])
        # plt.savefig(r'E:\ttune3\OneDrive - UW\Daniel_Group\cvae_test\stress.svg')
        # single = MultivariateNormal(loc = means_r2[:,-1,:], covariance_matrix=torch.diag_embed(vars_r2[:,-1,:]))
        
        # points_torch = np.zeros([n_samples,labels_.shape[1]])
        points_torch = []
        k = 0
        k_ = 0
        while k < n_samples:
        # while len(points_torch) < n_samples: 
            with torch.no_grad():
                means_r2, vars_r2, weights_r2 = model.eval_test_data(data_)
                mix = Categorical(probs = weights_r2)
                comp = MultivariateNormal(loc = means_r2, covariance_matrix=torch.diag_embed(vars_r2))
                gmm_r2 = MixtureSameFamily(mix, comp)
                
                # single = MultivariateNormal(loc = means_r2[:,0,:], 
                #                             covariance_matrix=torch.diag_embed(vars_r2[:,0,:]))
                
                # points = single.sample(torch.tensor([1000])).cpu().numpy().squeeze(-2)
                
                single_point = gmm_r2.sample().cpu().numpy() 
                if np.all(single_point <= training_dataset.tensors[1].cpu().numpy().max()*1.05) and np.all(single_point >= training_dataset.tensors[1].cpu().numpy().min()*1.05):
                # points = gmm_r2.sample(torch.tensor([1000])).cpu().numpy().squeeze(-2)
                    points_torch.extend(single_point)
                    k += 1
                else:
                    # pdb.set_trace()
                    # print('out')
                    k_ += 1
                if k_ > 2*n_samples:
                    break
                
                
        # print(k)
        print('points done, {} outside'.format(k_))
        points_torch = np.array(points_torch)    
        points = pd.DataFrame(points_torch)
        # points["id"] = str(i)  
        # points = points.head(8000)
        if points.shape[0]<8000:
            print(r'Warning: ' + str(points.shape[0]) + r'<8000 samples')


        kde = scipy.stats.gaussian_kde(points.T)

        def prob(X):
            return -kde.logpdf(X)
        
        # #estimate peak of distribution
        bandwidth = sklearn.cluster.estimate_bandwidth(points_torch, quantile=0.2, n_samples=2500)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(points_torch)
        cluster_centers = ms.cluster_centers_
        print(names[i] + ": " + repr(ms.cluster_centers_) + ": " + repr(kde.pdf(cluster_centers.T)))
        # print(names[i] + ": " + repr(cluster_centers[np.argmax(kde.evaluate(cluster_centers.T))]) + repr(prob(ms.cluster_centers_)))
        
        # cluster_centers = cluster_centers[np.argmax(kde.evaluate(cluster_centers.T))]
        
        
        
        
        # argmax
        cluster_centers = points_torch[np.argmax(kde.logpdf(points_torch.T))]
        cluster_centers = points_torch[torch.argmax(gmm_r2.log_prob(torch.tensor(points_torch, device=device)))]
        print(names[i] + ": " + repr(cluster_centers) + ": " + repr(kde.evaluate(cluster_centers)))

        
        

        # cluster_centers = scipy.optimize.differential_evolution(func = prob, bounds = bnds)
        cluster_centers = scipy.optimize.basinhopping(func = prob, x0 = cluster_centers, niter = 100, T = 1).x 
        print(names[i] + ": " + repr(cluster_centers) + ": " + repr(kde.evaluate(cluster_centers)))

        # cluster_centers = scipy.optimize.brute(func = prob, ranges = bnds)
        # (func = prob, x0 = cluster_centers, niter = 100, T = 5)
        
        
        
        # cluster_centers  = scipy.optimize.fmin(func=prob, x0=cluster_centers)
        # cluster_centers  = scipy.optimize.minimize(fun=prob, x0=cluster_centers, bounds = bnds)
        # print(names[i] + ": " + repr(cluster_centers.x))
        # cluster_centers = cluster_centers.x
        # print(names[i] + ": " + repr(cluster_centers.to_numpy()))
        
        points["id"] = str(names[i])
        list_of_dataframes.append(points)  
        
        cluster_centers_temp = pd.DataFrame(cluster_centers).T
        # cluster_centers_temp = pd.DataFrame([-0.60608235,   -0.39351082]).T
        cluster_centers_temp['id'] = str(names[i])
        cluster_centers_df.append(cluster_centers_temp)
        del cluster_centers_temp
        
        print("===============================================") 

# wt = pd.DataFrame(WT_best_label).T; wt['id'] = ['WT_sim']
# I61Q = pd.DataFrame(I61Q_best_label).T; I61Q['id'] = ['I61Q_sim']        

# cluster_centers_df.append(wt)
# cluster_centers_df.append(I61Q)
        
df = pd.concat(list_of_dataframes, axis=0, ignore_index=True)
cluster_centers_df = pd.concat(cluster_centers_df, axis=0, ignore_index=True)

# pair plot
g = sns.PairGrid(df, corner=True, hue='id')
color_for_trainingset = 'blue'
# color_for_trainingset = sns.color_palette('husl', 2) [-1] # this is the color from the question
# g.map_upper(sns.scatterplot, alpha=0.2, color=color_for_trainingset)
g.map_lower(sns.kdeplot, color=color_for_trainingset)
g.map_diag(sns.kdeplot, lw=3, color=color_for_trainingset)
g.add_legend()

g.data = pd.DataFrame(cluster_centers_df)
# g.data = data[data['type'] == 'Target 1']
# g.map_upper(sns.scatterplot, alpha=1, color='red')
g.map_lower(sns.scatterplot, alpha=1, color='green', zorder=30, s=500) 

# handles = [Line2D([], [], color='green', ls='', marker='o', markersize=50, label='most prob value'),
#            Line2D([], [], color=color_for_trainingset, lw=3, label='predicted dist')]

# g.add_legend(handles=handles)

for ax in g.axes.flat:
    try:
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.set_xlim((-1,2))
        # ax.set_xlim((training_dataset.tensors[1].min().cpu().numpy(), training_dataset.tensors[1].max().cpu().numpy()))
    except:
        pass
    try:
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.set_ylim((-1,2))
        # ax.set_ylim((training_dataset.tensors[1].min().cpu().numpy(), training_dataset.tensors[1].max().cpu().numpy()))
    except:
        pass


# name = r'E:\ttune3\OneDrive - UW\Daniel_Group\cvae_test\ave_twitch_cvae.svg'
# plt.savefig(name)
# plt.savefig(r'E:\ttune3\OneDrive - UW\Daniel_Group\cvae_test\exp_predictions_oct_26.svg') 
g.fig.suptitle(names[r[0]] + ' #{}'.format(i), fontsize = 40) 
plt.show()


#%%


from scipy.spatial import distance




distance.correlation()



# from sklearn.mixture import GaussianMixture


# X = df[df['id'] == 'WT_exp'].copy().drop('id', axis=1).T


# cov = np.cov(X)

# corr = np.corrcoef(X)




# gm_total = GaussianMixture(n_components=8, random_state=0).fit(X)




# ans = np.sum(gm_total.weights_[:,None,None] * gm_total.covariances_, 0) + np.sum(gm_total.weights_[:,None,None] * (gm_total.means_ - X.mean(axis=0).to_numpy()).T @ (gm_total.means_ - X.mean(axis=0).to_numpy()),0)

# dr = ans.copy(); dr[0,0]-=.01; dr[3,3]-=.01;



#%%


from scipy.optimize import minimize
from scipy import stats
from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np

class KDEDist(stats.rv_continuous):
    
    def __init__(self, kde, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kde = kde
    
    def _pdf(self, x):
        return self._kde.pdf(x)


# p = np.concatenate((np.random.normal(np.random.uniform(-10, 0), np.random.uniform(.5, 4), 2000),
#                     np.random.normal(np.random.uniform(-0, 10), np.random.uniform(.5, 4), 2000)))
# q = np.concatenate((np.random.normal(np.random.uniform(-10, 0), np.random.uniform(.5, 4), 2000),
#                     np.random.normal(np.random.uniform(-0, 10), np.random.uniform(.5, 4), 2000)))
x_ = np.linspace(-1, 2, 1000)
# plt.hist(p, bins=50)
# plt.hist(q, bins=50)

# filter(['Id', 'Rating2'])

# p = df[df['id'] == 'WT_exp'].copy().drop('id', axis=1)
p = df[df['id'] == 'WT_exp'].copy().filter([0], axis=1).to_numpy().flatten()

# q = df[df['id'] == 'I61Q_exp'].copy().drop('id', axis=1)
q = df[df['id'] == 'I61Q_exp'].copy().filter([0], axis=1).to_numpy().flatten()




corr = distance.correlation(p,q)


print(corr)


#%%




















def js_dist(x, x_, p, q, T=True):

    p = p + 10**x[1]
    
    p_kde = stats.gaussian_kde(10**p)
    Xp = KDEDist(p_kde)
    
    q_kde = stats.gaussian_kde(10**q)
    Xq = KDEDist(q_kde)
    
    ans = distance.jensenshannon(Xp.pdf(10**x_), Xq.pdf(10**x_))
    if T is True:
        return ans, Xp, Xq
    if T is False:
        return ans

a = 1
b = 0
x = [a,b]

ans, Xp, Xq = js_dist(x, x_, p, q, True)



fig, axe = plt.subplots()
# axe.hist(p, density=1, bins=50)
axe.plot(x_, Xp.pdf(x_), label = 'p+a')
axe.plot(x_, Xq.pdf(x_), label = 'q')
axe.legend()
# plt.hist(p+a, bins=20)
# plt.hist(q, bins=20)

print(ans)

x0 = np.array([1., 10**-1.])
res = minimize(js_dist, x0, method='nelder-mead', args=(x_, p, q, False),
               options={'xatol': 1e-8, 'disp': True, 'maxiter' : 100})

print(res.x)



ans, Xp, Xq = js_dist(res.x, x_, p, q, True)

fig, axe = plt.subplots()
# axe.hist(p, density=1, bins=50)
axe.plot(x_, (Xp.pdf(10**x_)), label = 'p+a')
axe.plot(x_, (Xq.pdf(10**x_)), label = 'q')
axe.legend()
# plt.hist(p+a, bins=20)
# plt.hist(q, bins=20)

print(ans)
























