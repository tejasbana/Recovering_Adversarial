from __future__ import print_function
import os
# os.system('nvidia-smi')
# os.system('pip3 install opendatasets --upgrade --quiet')
# os.system('pip3 install pytorch-lightning --quiet')

# os.system('gdown --id 129ruKo5e03Uwx7El5iZfXvbJVsojrav9')
# os.system('gdown --id 12L9rI_Iaz36ztXeBM5662IraTICOWHsV')
# os.system('gdown --id 1H-dhrytDv4MqUKTeE4ZCooHB2CwyMmZs')
# os.system('gdown --id 1TqaRC5FY0agW4ItuG34yq4kGFKhf6gqb')
# os.system('gdown --id 1Z1sG0dUuzutCBYoRP8bD3FXplK3RbSoV')
# os.system('gdown --id 1-3pW4BzMfaDRwvoqal6IMD6E95cDlwsS')
# os.system('gdown --id 1gKnaDTTRkeYabX5Ee3c7ngNutw_fpeu8')
# os.system('gdown --id 16J8a0Bl25u66RKKJKe9Ck1TytoQU5c3r')
# os.system('mkdir state_dicts')
# os.system('mv /content/inception_v3.pt /content/state_dicts/')
# os.system('mv /content/vgg13_bn.pt /content/state_dicts/')
# os.system('mv /content/resnet18.pt /content/state_dicts/')


# In[ ]:


import opendatasets as od
#od.download("https://www.kaggle.com/swaroopkml/cifar10-pngs-in-folders")
use_cuda=True

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
#import os
from tqdm.notebook import tqdm
from torchvision.utils import make_grid
from torchvision import models
from torch import Tensor
from torch.autograd import Variable
from torch.nn.utils import spectral_norm
import pickle

from models import resnet, vgg13, inception, Attention_block, SelfAttn, UNet, DNet
from attack import fgsm_attack, fgsm_attack_test, rfgsm_attack

batch_size_69=96
test_batch_size = 96
input_shape = (3, 32, 32)          # image size


E=[]
for i in range(1,33):
  E.append(i/255)
  
train_loader = torch.utils.data.DataLoader(
                ImageFolder('/content/cifar10-pngs-in-folders/cifar10/cifar10/train',
                        transform=transforms.Compose([ transforms.ToTensor() ])),
                        batch_size=batch_size_69,
                        shuffle=True,
                        num_workers=2,
                        pin_memory=True)

test_loader = torch.utils.data.DataLoader(
                ImageFolder('/content/cifar10-pngs-in-folders/cifar10/cifar10/test',
                        transform=transforms.Compose([ transforms.ToTensor()])),
                        batch_size=test_batch_size,
                        num_workers=2,
                        pin_memory=True)


def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        break

show_batch(train_loader)

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))



device = get_default_device()
discriminator_2 = to_device( inception(), device)
classifier = to_device(  vgg13(), device)
resnet_model = to_device( resnet(), device)
train_loader = DeviceDataLoader(train_loader, device)
test_loader =  DeviceDataLoader(test_loader, device)

@torch.no_grad()
def evaluate(model , val_loader, val=True):
    if val:
        model.eval()
    outputs = [model.validation_step(batch) for batch in tqdm(val_loader)]
    return model.validation_epoch_end(outputs)

print("discriminator_2 Train Accuracy: ",evaluate(discriminator_2 , train_loader, val=False) )
print("discriminator_2 Test  Accuracy: ",evaluate(discriminator_2 , test_loader))

print("classifier Train Accuracy : ",evaluate(classifier, train_loader, val=False))
print("classifier Test  Accuracy : ",evaluate(classifier, test_loader) )

print("resnet Train Accuracy : ",evaluate(resnet_model, train_loader, val=False))
print("resnet Test  Accuracy : ",evaluate(resnet_model, test_loader) )


# In[10]:


scaler = torch.cuda.amp.GradScaler()

# Train Inception Further
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []

    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                steps_per_epoch=len(train_loader))

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        lrs = []
        for batch in tqdm(train_loader):
            with torch.cuda.amp.autocast():
              loss = model.training_step(batch)
            train_losses.append(loss)
            scaler.scale(loss).backward()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            scaler.step(optimizer)

            # Updates the scale for next iteration
            scaler.update()
            optimizer.zero_grad()

            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history

#TODO: Check in models.py
classifier.eval()
discriminator_2.eval()
resnet_model.eval()

def init_weights(net, init='norm', gain=0.02):

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)

    net.apply(init_func)
    print(f"model initialized with {init} initialization")
    return net

def init_model(model, device):
    model = model.to(device)
    model = init_weights(model)
    return model


discriminator = init_model( DNet(input_shape) , device)
generator = init_model( UNet(True) , device)

#GAN training
scaler = torch.cuda.amp.GradScaler()

criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()
criterion_classification = nn.CrossEntropyLoss()
g_lambda = 100

def train(perturbed_data , data , target , lr):
  g_optimizer = torch.optim.Adam(generator.parameters() , betas=(0.5,0.999) , lr = lr)
  d_optimizer = torch.optim.Adam(discriminator.parameters(),betas=(0.5,0.999), lr = lr )
  valid = Variable(Tensor(np.ones((data.size(0), *discriminator.output_shape))), requires_grad=False).cuda()
  fake = Variable(Tensor(np.zeros((data.size(0), *discriminator.output_shape))), requires_grad=False).cuda()

  batch_size = data.shape[0]
  # fake_image = perturbed_data - generator(perturbed_data)
  with torch.cuda.amp.autocast():
    fake_image = generator(perturbed_data)
    mean = torch.Tensor([0.5])
    fake_image = fake_image/2
    fake_image = fake_image + mean.expand_as(fake_image).cuda()
  # fake_image = perturbed_data - fake_image

  # Train the discriminator. The loss would be the sum of the losses over
  d_optimizer.zero_grad()
  d_loss = 0

  with torch.cuda.amp.autocast():
    d_real_loss = criterion_GAN(discriminator(data), valid )
    d_fake_loss = criterion_GAN(discriminator(fake_image.detach()) , fake )
    d_loss = d_real_loss + d_fake_loss

  scaler.scale(d_loss).backward()
  # d_loss.backward(retain_graph=True)
  scaler.step(d_optimizer)

  # Train the generator. The loss would be the sum of the adversarial loss
  # due to the GAN and L1 distance loss between the fake and target images.

  g_optimizer.zero_grad()
  g_loss = 0

  with torch.cuda.amp.autocast():
    loss_GAN = criterion_GAN(discriminator(fake_image) , valid)
    # loss_content = criterion_content( discriminator_2.feature_extractor(fake_image), discriminator_2.feature_extractor(data).detach())

    fake_out = discriminator_2(fake_image) # generator loss 1
    loss_classification = criterion_classification(fake_out , target)

    loss_content = criterion_content(fake_image , data)
    loss_content = g_lambda * loss_content

    g_loss =  loss_GAN + loss_classification + loss_content

  scaler.scale(g_loss).backward()
  scaler.step(g_optimizer)
  scaler.update()

  return fake_image.detach(), d_loss.detach() ,g_loss.detach(), loss_GAN.detach() , loss_content.detach()


def training(epochs,lr, attacking_model_1, attacking_model_2, device, test_loader ):

    losses_g = []
    losses_d = []
    losses_g2 = []
    losses_md = []
    # Accuracy counter
    adv_examples = []
    start_idx = 0
    # Loop over all examples in test set
    for epoch in range(epochs):
      generator.train()
      discriminator.train()
      discriminator_2.eval()
      correct = 0
      total = 0
      d_running_loss = 0.0
      g_running_loss = 0.0
      md_running_loss = 0.0
      g_running_loss2 = 0.0
      g_content_loss = 0.0
      epsilon = E
      number = 1
      if epoch > 25 and lr > 2e-5:
        lr = lr - 1e-5
      # else if epoch > 20:
      #   lr = lr - 1e-
      # else if epoch > 5:
      #   lr = lr - 1e-5
      for batch in tqdm(test_loader):
        data, target = batch
        start_idx = start_idx+1
        batch_size = int(data.shape[0] / 6 )
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True
        # Forward pass the data through the model

        with torch.cuda.amp.autocast():
          output = attacking_model_1(data)
          init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
          # Calculate the loss
          loss = F.nll_loss(output, target)
        # Zero all existing gradients
        attacking_model_1.zero_grad()
        # Calculate gradients of model in backward pass
        scaler.scale(loss).backward()
        # Collect datagrad
        data_grad = data.grad.data[:batch_size]

        # Call FGSM Attack 1
        perturbed_data = fgsm_attack(data[:batch_size], epsilon, data_grad)

        data.grad.zero_()

        with torch.cuda.amp.autocast():
          output = attacking_model_2(data)
          init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
          loss = F.nll_loss(output, target)

        attacking_model_2.zero_grad()
        scaler.scale(loss).backward()
        data_grad = data.grad.data[batch_size:batch_size*2]

        # Call FGSM Attack 2

        perturbed_data = torch.cat((perturbed_data, fgsm_attack(data[batch_size:batch_size*2], epsilon, data_grad)), 0)


        #model 3
        data.grad.zero_()
        with torch.cuda.amp.autocast():
          output = resnet_model(data)
          init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
          loss = F.nll_loss(output, target)
        resnet_model.zero_grad()
        scaler.scale(loss).backward()
        data_grad = data.grad.data[batch_size*2:batch_size*3]

        # Call FGSM Attack 3
        perturbed_data = torch.cat((perturbed_data, fgsm_attack(data[batch_size*2:batch_size*3], epsilon, data_grad)), 0)

        # Call RFGSM
        alpha = 4/255
        data.grad.zero_()
        adv_images = data.clone().detach() + alpha*torch.randn_like(data).sign()
        adv_images.requires_grad = True
        #model 1
        with torch.cuda.amp.autocast():
          output = attacking_model_1(adv_images)
          init_pred = output.max(1, keepdim=True)[1]
          loss = F.nll_loss(output, target)
        attacking_model_1.zero_grad()
        scaler.scale(loss).backward()
        data_grad = adv_images.grad.data[batch_size*3:batch_size*4]
        perturbed_data = torch.cat((perturbed_data, rfgsm_attack(adv_images[batch_size*3:batch_size*4], epsilon, data_grad , alpha) ), 0)

        # Call RFGSM
        data.grad.zero_()
        adv_images = data.clone().detach() + alpha*torch.randn_like(data).sign()
        adv_images.requires_grad = True
        #Model 2
        with torch.cuda.amp.autocast():
          output = attacking_model_2(adv_images)
          init_pred = output.max(1, keepdim=True)[1]
          loss = F.nll_loss(output, target)
        attacking_model_2.zero_grad()
        scaler.scale(loss).backward()
        data_grad = adv_images.grad.data[batch_size*4:batch_size*5]
        perturbed_data = torch.cat((perturbed_data, rfgsm_attack(adv_images[batch_size*4:batch_size*5], epsilon, data_grad , alpha) ), 0)

        # Call RFGSM
        data.grad.zero_()
        adv_images = data.clone().detach() + alpha*torch.randn_like(data).sign()
        adv_images.requires_grad = True
        #Model 2
        with torch.cuda.amp.autocast():
          output = resnet_model(adv_images)
          init_pred = output.max(1, keepdim=True)[1]
          loss = F.nll_loss(output, target)
        resnet_model.zero_grad()
        scaler.scale(loss).backward()
        data_grad = adv_images.grad.data[batch_size*5:]
        perturbed_data = torch.cat((perturbed_data, rfgsm_attack(adv_images[batch_size*5:], epsilon, data_grad , alpha) ), 0)

        data.grad.zero_()
        data.requires_grad = False

        if(start_idx < 20):
           save_adv(start_idx , perturbed_data)

        #Training GANs
        gen_image , d_loss , g_loss  , g_loss2, g_content = train(perturbed_data , data , target , lr)
        d_running_loss += d_loss
        g_running_loss += g_loss
        g_running_loss2 += g_loss2
        g_content_loss += g_content

        # Re-classify the perturbed image
        output = attacking_model_1(gen_image)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        final_pred.squeeze(1)
        for indx in range(len(final_pred)):
            total += 1
            if final_pred[indx].item() == target[indx].item():
                correct += 1

    # Calculate final accuracy for this epsilon
      length_data = float(len(test_loader))
      print('Epoch : {} ,lr : {}, G_loss_total : {:.4f} , D_loss_total : {:.4f} , Gen_D_loss : {:.4f}, Gen_content_loss : {:.4f} , Gen_classification_loss : {:.4f}'.format(epoch, lr,g_running_loss/length_data, d_running_loss/length_data, g_running_loss2/length_data, g_content_loss/length_data,(g_running_loss-g_running_loss2-g_content_loss)/length_data ))
      var = total #float(sum([len(itm[1]) for itm in train_loader]))
      final_acc = correct/var
      print("Training Accuracy = {} / {} = {}".format(correct, var, final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples



def testing_models(attacking_model, target_model, device, test_loader, epsilon, model_name ):

    # Accuracy counter
    generator.eval()
    correct = 0
    original_examples = []
    attacked_examples = []
    adv_examples = []
    start_idx = 0
    total = 0
    # Loop over all examples in test set
    for data, target in test_loader:
        start_idx = start_idx+1
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = attacking_model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        # if init_pred.item() != target.item():
        #     continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        attacking_model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack_test(data, epsilon, data_grad)
        #save_samples(start_idx, perturbed_data)
        gen_image = generator(perturbed_data)
        mean = torch.Tensor([0.5])
        gen_image = gen_image/2
        gen_image = gen_image + mean.expand_as(gen_image).cuda()
        # gen_image = perturbed_data - gen_image

        if(start_idx < 20):
           save_samples(start_idx , gen_image)
        # Re-classify the perturbed image
        output = target_model(gen_image)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        final_pred.squeeze(1)
        for indx in range(len(final_pred)):
            total += 1
            if final_pred[indx].item() == target[indx].item():
                correct += 1
                # Save some adv examples for visualization later
                if len(adv_examples) < 10:
                    adv_ex = gen_image[indx].squeeze().detach().cpu().numpy()
                    adv_examples.append( (init_pred[indx].item(), final_pred[indx].item(), adv_ex) )

                    adv_ex = data[indx].squeeze().detach().cpu().numpy()
                    original_examples.append( (init_pred[indx].item(), final_pred[indx].item(), adv_ex) )

                    adv_ex = perturbed_data[indx].squeeze().detach().cpu().numpy()
                    attacked_examples.append( (init_pred[indx].item(), final_pred[indx].item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    var = total #float(sum([len(itm[1]) for itm in test_loader]))
    final_acc = correct/var
    print("Epsilon: {}\tTest Accuracy by {} = {} / {} = {}".format(epsilon*255,model_name, correct, var, final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples , original_examples , attacked_examples



def testing_all(target_model, device, test_loader, epsilon, model_name ):

    # Accuracy counter
    generator.eval()
    correct = 0
    original_examples = []
    attacked_examples = []
    adv_examples = []
    epsilon = E
    start_idx = 0
    total = 0
    # Loop over all examples in test set
    for batch in test_loader:
        data, target = batch
        start_idx = start_idx+1
        batch_size = int(data.shape[0] / 6 )
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True
        # Forward pass the data through the model
        output = discriminator_2(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        # Calculate the loss
        loss = F.nll_loss(output, target)
        # Zero all existing gradients
        discriminator_2.zero_grad()
        # Calculate gradients of model in backward pass
        loss.backward()
        # Collect datagrad
        data_grad = data.grad.data[:batch_size]

        # Call FGSM Attack 1
        perturbed_data = fgsm_attack(data[:batch_size], epsilon, data_grad)

        data.grad.zero_()
        output = classifier(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        loss = F.nll_loss(output, target)
        classifier.zero_grad()
        loss.backward()
        data_grad = data.grad.data[batch_size:batch_size*2]

        # Call FGSM Attack 2

        perturbed_data = torch.cat((perturbed_data, fgsm_attack(data[batch_size:batch_size*2], epsilon, data_grad)), 0)


        #model 3
        data.grad.zero_()
        output = resnet_model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        loss = F.nll_loss(output, target)
        resnet_model.zero_grad()
        loss.backward()
        data_grad = data.grad.data[batch_size*2:batch_size*3]

        # Call FGSM Attack 3
        perturbed_data = torch.cat((perturbed_data, fgsm_attack(data[batch_size*2:batch_size*3], epsilon, data_grad)), 0)

        # Call RFGSM
        alpha = 4/255
        data.grad.zero_()
        adv_images = data.clone().detach() + alpha*torch.randn_like(data).sign()
        adv_images.requires_grad = True
        #model 1
        output = discriminator_2(adv_images)
        init_pred = output.max(1, keepdim=True)[1]
        loss = F.nll_loss(output, target)
        discriminator_2.zero_grad()
        loss.backward()
        data_grad = adv_images.grad.data[batch_size*3:batch_size*4]
        perturbed_data = torch.cat((perturbed_data, rfgsm_attack(adv_images[batch_size*3:batch_size*4], epsilon, data_grad , alpha) ), 0)

        # Call RFGSM
        data.grad.zero_()
        adv_images = data.clone().detach() + alpha*torch.randn_like(data).sign()
        adv_images.requires_grad = True
        #Model 2
        output = classifier(adv_images)
        init_pred = output.max(1, keepdim=True)[1]
        loss = F.nll_loss(output, target)
        classifier.zero_grad()
        loss.backward()
        data_grad = adv_images.grad.data[batch_size*4:batch_size*5]
        perturbed_data = torch.cat((perturbed_data, rfgsm_attack(adv_images[batch_size*4:batch_size*5], epsilon, data_grad , alpha) ), 0)

        # Call RFGSM
        data.grad.zero_()
        adv_images = data.clone().detach() + alpha*torch.randn_like(data).sign()
        adv_images.requires_grad = True
        #Model 2
        output = resnet_model(adv_images)
        init_pred = output.max(1, keepdim=True)[1]
        loss = F.nll_loss(output, target)
        resnet_model.zero_grad()
        loss.backward()
        data_grad = adv_images.grad.data[batch_size*5:]
        perturbed_data = torch.cat((perturbed_data, rfgsm_attack(adv_images[batch_size*5:], epsilon, data_grad , alpha) ), 0)

        data.grad.zero_()
        data.requires_grad = False
        if(start_idx < 20):
           save_adv(start_idx , perturbed_data)

        gen_image = generator(perturbed_data)
        mean = torch.Tensor([0.5])
        gen_image = gen_image/2
        gen_image = gen_image + mean.expand_as(gen_image).cuda()
        # gen_image = perturbed_data - gen_image

        if(start_idx < 20):
           save_samples(start_idx , gen_image)
        # Re-classify the perturbed image
        output = target_model(gen_image)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        final_pred.squeeze(1)
        for indx in range(len(final_pred)):
            total += 1
            if final_pred[indx].item() == target[indx].item():
                correct += 1
                # Save some adv examples for visualization later
                # if len(adv_examples) < 10:
                #     adv_ex = gen_image[indx].squeeze().detach().cpu().numpy()
                #     adv_examples.append( (init_pred[indx].item(), final_pred[indx].item(), adv_ex) )

                #     adv_ex = data[indx].squeeze().detach().cpu().numpy()
                #     original_examples.append( (init_pred[indx].item(), final_pred[indx].item(), adv_ex) )

                #     adv_ex = perturbed_data[indx].squeeze().detach().cpu().numpy()
                #     attacked_examples.append( (init_pred[indx].item(), final_pred[indx].item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    var = total #float(sum([len(itm[1]) for itm in test_loader]))
    final_acc = correct/var
    print("All attack Test Accuracy by {} = {} / {} = {}".format(model_name, correct, var, final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples , original_examples , attacked_examples

#Extra Testing Code

def testing_without_gan(model, device, test_loader, epsilon):
    model.eval()
    # Accuracy counter
    generator.eval()
    correct = 0
    original_examples = []
    attacked_examples = []
    adv_examples = []
    start_idx = 0
    # Loop over all examples in test set
    for data, target in tqdm(test_loader):
        start_idx = start_idx+1
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        # if init_pred.item() != target.item():
        #     continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        gen_image = fgsm_attack_test(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(gen_image)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        final_pred.squeeze(1)
        for indx in range(len(final_pred)):
            if final_pred[indx].item() == target[indx].item():
                correct += 1

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(sum([len(itm[1]) for itm in test_loader]))
    print("Epsilon: {}\tTest Accuracy by Resnet-18 = {} / {} = {}".format(epsilon, correct, sum([len(itm[1]) for itm in test_loader]), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc


def testing_vgg( vgg, device, test_loader, epsilon ):

    # Accuracy counter
    generator.eval()
    correct = 0
    original_examples = []
    attacked_examples = []
    adv_examples = []
    start_idx = 0
    # Loop over all examples in test set
    for data, target in test_loader:
        start_idx = start_idx+1
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = classifier(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        classifier.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack_test(data, epsilon, data_grad)
        #save_samples(start_idx, perturbed_data)
        gen_image = generator(perturbed_data)
        #generated image output in 0 to 1 range
        mean = torch.tensor(0.5)
        gen_image = gen_image/2
        gen_image = gen_image + mean.expand_as(gen_image).cuda()

        if(start_idx < 20):
           save_samples(start_idx , gen_image)
        # Re-classify the perturbed image
        output = vgg(gen_image)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        final_pred.squeeze(1)
        for indx in range(len(final_pred)):
            if final_pred[indx].item() == target[indx].item():
                correct += 1
                # Save some adv examples for visualization later
                if len(adv_examples) < 10:
                    adv_ex = gen_image[indx].squeeze().detach().cpu().numpy()
                    adv_examples.append( (init_pred[indx].item(), final_pred[indx].item(), adv_ex) )

                    adv_ex = data[indx].squeeze().detach().cpu().numpy()
                    original_examples.append( (init_pred[indx].item(), final_pred[indx].item(), adv_ex) )

                    adv_ex = perturbed_data[indx].squeeze().detach().cpu().numpy()
                    attacked_examples.append( (init_pred[indx].item(), final_pred[indx].item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(sum([len(itm[1]) for itm in test_loader]))
    print("Epsilon: {}\tTest Accuracy by vgg-13 = {} / {} = {}".format(epsilon*255, correct, sum([len(itm[1]) for itm in test_loader]), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples , original_examples , attacked_examples

#Fit

os.system('mkdir generated')
os.system('mkdir adversarial_sample')



sample_dir = 'generated/'
def save_samples(index, fake_images, show=True):
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    save_image(fake_images, os.path.join(sample_dir, fake_fname))

adv_dir = 'adversarial_sample/'
def save_adv(index, fake_images, show=True):
    fake_fname = 'adv-images-{0:0=4d}.png'.format(index)
    save_image(fake_images, os.path.join(adv_dir, fake_fname))


# In[ ]:


# generator.load_state_dict(torch.load('load_now.pth'))
# generator.load_state_dict( torch.load('Gen_30.pth'))
# discriminator.load_state_dict( torch.load('Disc_30.pth'))


# In[ ]:


# acc, ex = training(25,2e-4 ,discriminator_2, classifier,  device, train_loader)
# acc, ex = training(15,5e-5 ,discriminator_2, classifier,  device, train_loader)


# In[29]:


#RANDOM
epsilons = [0 ,1/255,2/255, 4/255,8/255  ,16/255 , 32/255]
accuracies2 = []
examples2 = []
test_accuracies2 = []
test_examples2 = []
original_examples2 = []
attacked_examples2 = []

# Run train for each epsilon
acc, ex = training(50,2e-4 ,discriminator_2, classifier,  device, train_loader)
# for eps in epsilons:
#     print("######################################################")
#     test_acc, test_ex ,ori_ex ,attack_ex= testing_models(discriminator_2, discriminator_2, device, test_loader, eps, "Inception_Dis")
#     test_accuracies2.append(test_acc)
#     test_examples2.append(test_ex)
#     original_examples2.append(ori_ex)
#     attacked_examples2.append(attack_ex)
#     test_acc, test_ex ,ori_ex ,attack_ex = testing_models(discriminator_2, classifier, device, test_loader, eps, "VGG_robustness")
#     # test_acc, test_ex ,ori_ex ,attack_ex = testing_vgg(inception_model, device, test_loader, eps)



testing_all(discriminator_2, device, test_loader, eps, "Inception_Dis")
testing_all(classifier, device, test_loader, eps, "VGG_robustness")

for eps in epsilons:
    print("######################################################")
    test_acc, test_ex ,ori_ex ,attack_ex= testing_models(resnet_model, discriminator_2, device, test_loader, eps, "Inception_Dis")
    test_acc, test_ex ,ori_ex ,attack_ex = testing_models(resnet_model, classifier, device, test_loader, eps, "VGG_robustness")

#Post training

scaler = torch.cuda.amp.GradScaler()
def post_train_generator(perturbed_data , data , target,lr,g_lambda = 10 , loss_unchange=True):

  smooth = 0
  #data range from 0 to 1
  optimizer = torch.optim.Adam(generator.parameters() , betas=(0.5,0.999) , lr = lr)
  criterion = nn.L1Loss()

  batch_size = data.shape[0]
  # fake images are generated by passing them through the generator.
  optimizer.zero_grad()

  with torch.cuda.amp.autocast():
    fake_image = generator(perturbed_data)
    mean = torch.Tensor([0.5])
    fake_image = fake_image/2
    fake_image = fake_image + mean.expand_as(fake_image).cuda()
    # fake_image = perturbed_data - fake_image

    g_image_distance_loss = 0 #g_lambda * criterion(fake_image , data)         # generator loss 2

    for f1, f2 in zip(discriminator_2.feature_extractor(fake_image), discriminator_2.feature_extractor(data)):
      # Compute content loss with target and content images
      g_image_distance_loss += torch.mean(torch.abs(f1 - f2))

    g_image_distance_loss = g_lambda * g_image_distance_loss

  # g_image_distance_loss.backward(retain_graph=True)
  scaler.scale(g_image_distance_loss).backward()
  scaler.step(optimizer)
  scaler.update()

  return fake_image , g_image_distance_loss.detach()


def post_train(epochs, attacking_model_1, attacking_model_2, device, test_loader,lr ,g_lambda,second_loss=True):

    losses_g = []
    losses_d = []
    losses_g2 = []
    losses_md = []
    # Accuracy counter
    adv_examples = []
    start_idx = 0
    # Loop over all examples in test set
    for epoch in range(epochs):
      generator.train()
      correct = 0
      total = 0
      g_running_loss = 0.0
      epsilon = E
      number = 1
      if epoch > 5 and lr > 2e-5:
        lr = lr - 1e-5
      for batch in tqdm(test_loader):
        data, target = batch
        start_idx = start_idx+1
        batch_size = int(data.shape[0] / 6 )
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True
        # Forward pass the data through the model
        with torch.cuda.amp.autocast():
          output = attacking_model_1(data)
          init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
          # Calculate the loss
          loss = F.nll_loss(output, target)
        # Zero all existing gradients
        attacking_model_1.zero_grad()
        # Calculate gradients of model in backward pass
        scaler.scale(loss).backward()
        # Collect datagrad
        data_grad = data.grad.data[:batch_size]

        # Call FGSM Attack 1
        perturbed_data = fgsm_attack(data[:batch_size], epsilon, data_grad)

        data.grad.zero_()

        with torch.cuda.amp.autocast():
          output = attacking_model_2(data)
          init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
          loss = F.nll_loss(output, target)

        attacking_model_2.zero_grad()
        scaler.scale(loss).backward()
        data_grad = data.grad.data[batch_size:batch_size*2]

        # Call FGSM Attack 2

        perturbed_data = torch.cat((perturbed_data, fgsm_attack(data[batch_size:batch_size*2], epsilon, data_grad)), 0)


        #model 3
        data.grad.zero_()
        with torch.cuda.amp.autocast():
          output = resnet_model(data)
          init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
          loss = F.nll_loss(output, target)
        resnet_model.zero_grad()
        scaler.scale(loss).backward()
        data_grad = data.grad.data[batch_size*2:batch_size*3]

        # Call FGSM Attack 3
        perturbed_data = torch.cat((perturbed_data, fgsm_attack(data[batch_size*2:batch_size*3], epsilon, data_grad)), 0)

        # Call RFGSM
        alpha = 4/255
        data.grad.zero_()
        adv_images = data.clone().detach() + alpha*torch.randn_like(data).sign()
        adv_images.requires_grad = True
        #model 1
        with torch.cuda.amp.autocast():
          output = attacking_model_1(adv_images)
          init_pred = output.max(1, keepdim=True)[1]
          loss = F.nll_loss(output, target)
        attacking_model_1.zero_grad()
        scaler.scale(loss).backward()
        data_grad = adv_images.grad.data[batch_size*3:batch_size*4]
        perturbed_data = torch.cat((perturbed_data, rfgsm_attack(adv_images[batch_size*3:batch_size*4], epsilon, data_grad , alpha) ), 0)

        # Call RFGSM
        data.grad.zero_()
        adv_images = data.clone().detach() + alpha*torch.randn_like(data).sign()
        adv_images.requires_grad = True
        #Model 2
        with torch.cuda.amp.autocast():
          output = attacking_model_2(adv_images)
          init_pred = output.max(1, keepdim=True)[1]
          loss = F.nll_loss(output, target)
        attacking_model_2.zero_grad()
        scaler.scale(loss).backward()
        data_grad = adv_images.grad.data[batch_size*4:batch_size*5]
        perturbed_data = torch.cat((perturbed_data, rfgsm_attack(adv_images[batch_size*4:batch_size*5], epsilon, data_grad , alpha) ), 0)

        # Call RFGSM
        data.grad.zero_()
        adv_images = data.clone().detach() + alpha*torch.randn_like(data).sign()
        adv_images.requires_grad = True
        #Model 2
        with torch.cuda.amp.autocast():
          output = resnet_model(adv_images)
          init_pred = output.max(1, keepdim=True)[1]
          loss = F.nll_loss(output, target)
        resnet_model.zero_grad()
        scaler.scale(loss).backward()
        data_grad = adv_images.grad.data[batch_size*5:]
        perturbed_data = torch.cat((perturbed_data, rfgsm_attack(adv_images[batch_size*5:], epsilon, data_grad , alpha) ), 0)

        data.grad.zero_()
        data.requires_grad = False
        #Training GANs
        gen_image , g_loss = post_train_generator(perturbed_data , data , target,lr,g_lambda,second_loss)

        g_running_loss += g_loss
        # Re-classify the perturbed image
        output = attacking_model_1(gen_image)

        if(start_idx < 20):
           save_adv(start_idx , perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        final_pred.squeeze(1)
        # correct = torch.sum(final_pred == target).item() / len(final_pred)
        for indx in range(len(final_pred)):
            total += 1
            if final_pred[indx].item() == target[indx].item():
                correct += 1
                # Special case for saving 0 epsilon examples
                if (epsilon == 0) and (len(adv_examples) < 5):
                    adv_ex = gen_image[indx].squeeze().detach().cpu().numpy()
                    adv_examples.append( (init_pred[indx].item(), final_pred[indx].item(), adv_ex) )
            else:
                # Save some adv examples for visualization later
                if len(adv_examples) < 5:
                    adv_ex = gen_image[indx].squeeze().detach().cpu().numpy()
                    adv_examples.append( (init_pred[indx].item(), final_pred[indx].item(), adv_ex) )

    # Calculate final accuracy for this epsilon
      print('Epoch : {} , g_epoch_loss : {:.4f} '.format(epoch,g_running_loss/float(len(test_loader))))
      var = total #float(sum([len(itm[1]) for itm in train_loader]))
      final_acc = correct/var
      print("Training Accuracy = {} / {} = {}".format( correct, var, final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

# pre_acc, pre_ex = post_train(5 ,discriminator_2, classifier, device, train_loader,lr=1e-4,g_lambda = 10,second_loss=True)
# pre_acc, pre_ex = post_train(10 ,discriminator_2, classifier, device, train_loader,lr=5e-5,g_lambda = 10,second_loss=True)
# pre_acc, pre_ex = post_train(20 ,discriminator_2, classifier, device, train_loader,lr=1e-5,g_lambda = 10,second_loss=False)

pre_acc, pre_ex = post_train(20 ,discriminator_2, classifier, device, train_loader,lr=1e-4,g_lambda = 10,second_loss=True)

#RANDOM
epsilons = [0 ,1/255,2/255, 4/255,8/255  ,16/255 , 32/255]
accuracies2 = []
examples2 = []
test_accuracies2 = []
test_examples2 = []
original_examples2 = []
attacked_examples2 = []

for eps in epsilons:
    print("######################################################")
    test_acc, test_ex ,ori_ex ,attack_ex= testing_models(discriminator_2, discriminator_2, device, test_loader, eps, "Inception_Dis")
    test_accuracies2.append(test_acc)
    test_examples2.append(test_ex)
    original_examples2.append(ori_ex)
    attacked_examples2.append(attack_ex)
    test_acc, test_ex ,ori_ex ,attack_ex = testing_models(discriminator_2, classifier, device, test_loader, eps, "VGG_robustness")
    # test_acc, test_ex ,ori_ex ,attack_ex = testing_vgg(inception_model, device, test_loader, eps)

# pre_acc, pre_ex = post_train(10 ,discriminator_2, classifier, device, train_loader,lr=1e-6,g_lambda = 100,second_loss=False)

# for eps in epsilons:
#     print("######################################################")
#     test_acc, test_ex ,ori_ex ,attack_ex= testing_models(discriminator_2, discriminator_2, device, test_loader, eps, "Inception_Dis")
#     test_accuracies2.append(test_acc)
#     test_examples2.append(test_ex)
#     original_examples2.append(ori_ex)
#     attacked_examples2.append(attack_ex)
#     test_acc, test_ex ,ori_ex ,attack_ex = testing_models(discriminator_2, classifier, device, test_loader, eps, "VGG_robustness")
#     # test_acc, test_ex ,ori_ex ,attack_ex = testing_vgg(inception_model, device, test_loader, eps)


testing_all(discriminator_2, device, test_loader, eps, "Inception_Dis")
testing_all(classifier, device, test_loader, eps, "VGG_robustness")

for eps in epsilons:
    print("######################################################")
    test_acc, test_ex ,ori_ex ,attack_ex= testing_models(resnet_model, discriminator_2, device, test_loader, eps, "Inception_Dis")
    test_acc, test_ex ,ori_ex ,attack_ex = testing_models(resnet_model, classifier, device, test_loader, eps, "VGG_robustness")


pre_acc, pre_ex = post_train(3 ,discriminator_2, classifier, device, train_loader,lr=5e-6,g_lambda = 10,second_loss=True)
pre_acc, pre_ex = post_train(2 ,discriminator_2, classifier, device, train_loader,lr=1e-6,g_lambda = 10,second_loss=True)

for eps in epsilons:
    print("######################################################")
    test_acc, test_ex ,ori_ex ,attack_ex= testing_models(discriminator_2, discriminator_2, device, test_loader, eps, "Inception_Dis")
    test_accuracies2.append(test_acc)
    test_examples2.append(test_ex)
    original_examples2.append(ori_ex)
    attacked_examples2.append(attack_ex)
    test_acc, test_ex ,ori_ex ,attack_ex = testing_models(discriminator_2, classifier, device, test_loader, eps, "VGG_robustness")
    # test_acc, test_ex ,ori_ex ,attack_ex = testing_vgg(inception_model, device, test_loader, eps)
