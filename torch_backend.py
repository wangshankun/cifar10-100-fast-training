import numpy as np
import torch
from torch import nn
import torchvision
from core import  cat, to_numpy

torch.backends.cudnn.benchmark = True
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@cat.register(torch.Tensor)
def _(*xs):
    return torch.cat(xs)

@to_numpy.register(torch.Tensor)
def _(x):
    return x.detach().cpu().numpy()  

def warmup_cudnn(model, batch_size):
    #run forward and backward pass of the model on a batch of random inputs
    #to allow benchmarking of cudnn kernels 
    batch = {
        'input': torch.Tensor(np.random.rand(batch_size,3,32,32)).cuda().half(), 
        'target': torch.LongTensor(np.random.randint(0,10,batch_size)).cuda()
    }
    model.train(True)
    o = model(batch)
    o['loss'].sum().backward()
    model.zero_grad()
    torch.cuda.synchronize()


#####################
## dataset
#####################

def cifar10(root):
    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    return {
       'train': {'data': train_set.data, 'labels': train_set.targets},
       'test': {'data': test_set.data, 'labels': test_set.targets}      
    }

def cifar100(root):
    train_set = torchvision.datasets.CIFAR100(root=root, train=True, download=True)
    test_set = torchvision.datasets.CIFAR100(root=root, train=False, download=True)
    return {
       'train': {'data': train_set.data, 'labels': train_set.targets},
       'test': {'data': test_set.data, 'labels': test_set.targets}      
    }


#####################
## data loading
#####################

class Batches():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False,scale_factor=1.0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=False, shuffle=shuffle, drop_last=drop_last
        )
        #self.resize_func = torch.nn.Upsample(scale_factor=scale_factor, mode='bilinear')
    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices()
            
        return ({'input': x.cuda().half()
                 ,'target': y.cuda().long()} for (x,y) in self.dataloader)
    
    def __len__(self): 
        return len(self.dataloader)

#####################
## torch stuff
#####################

class Identity(nn.Module):
    def forward(self, x): return x
    
class Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
    def __call__(self, x): 
        return x*self.weight
    
class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), x.size(1))

class Add(nn.Module):
    def forward(self, x, y): return x + y 
    
class Concat(nn.Module):
    def forward(self, *xs): return torch.cat(xs, 1)
    
class Correct(nn.Module):
    def forward(self, classifier, target):
        return classifier.max(dim = 1)[1] == target

def batch_norm(num_channels, bn_bias_init=None, bn_bias_freeze=False, bn_weight_init=None, bn_weight_freeze=False):
    m = nn.BatchNorm2d(num_channels)
    if bn_bias_init is not None:
        m.bias.data.fill_(bn_bias_init)
    if bn_bias_freeze:
        m.bias.requires_grad = False
    if bn_weight_init is not None:
        m.weight.data.fill_(bn_weight_init)
    if bn_weight_freeze:
        m.weight.requires_grad = False
        
    return m

trainable_params = lambda model:filter(lambda p: p.requires_grad, model.parameters())

class TorchOptimiser():
    def __init__(self, weights, optimizer, step_number=0, **opt_params):
        self.weights = weights
        self.step_number = step_number
        self.opt_params = opt_params
        self._opt = optimizer(weights, **self.param_values())
    
    def param_values(self):
        return {k: v(self.step_number) if callable(v) else v for k,v in self.opt_params.items()}
    
    def step(self):
        self.step_number += 1
        self._opt.param_groups[0].update(**self.param_values())
        self._opt.step()

    def __repr__(self):
        return repr(self._opt)
        
def SGD(weights, lr=0, momentum=0, weight_decay=0, dampening=0, nesterov=False):
    return TorchOptimiser(weights, torch.optim.SGD, lr=lr, momentum=momentum, 
                          weight_decay=weight_decay, dampening=dampening, 
    nesterov=nesterov)
