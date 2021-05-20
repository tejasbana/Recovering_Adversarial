from __future__ import print_function
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
import os
from tqdm.notebook import tqdm
from torchvision.utils import make_grid
from torchvision import models
from torch import Tensor
from torch.autograd import Variable
from torch.nn.utils import spectral_norm
import pickle

E=[]
for i in range(1,33):
  E.append(i/255)


# In[21]:


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image.clone() #+ epsilon[0]*sign_data_grad

    sample_per_class = 1
    length_epsilon = int(image.shape[0] / sample_per_class)

    for index_e in range(length_epsilon):
      index = sample_per_class*index_e
      perturbed_image[index:(index+sample_per_class)]   = image[index:(index+sample_per_class)]   + epsilon[index_e]*sign_data_grad[index:(index+sample_per_class)]


    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


# In[22]:


def fgsm_attack_test(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


# In[23]:


def rfgsm_attack(adv_image, epsilon, data_grad , alpha):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = adv_image.clone() #+ epsilon[0]*sign_data_grad

    sample_per_class = 1
    length_epsilon = int(adv_image.shape[0] / sample_per_class)

    for index_e in range(length_epsilon):
      index = sample_per_class*index_e
      perturbed_image[index:(index+sample_per_class)]  = adv_image[index:(index+sample_per_class)]   + (epsilon[index_e] - alpha)* sign_data_grad[index:(index+sample_per_class)]


    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image
