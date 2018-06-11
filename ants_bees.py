from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
import time
import os
import copy
import getopt, sys

data = 'hymenoptera_data'
transforms = transforms.Compose([
  transforms.Resize(224),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img_datasets = { 
  'train': datasets.ImageFolder(os.path.join(data, 'train'), transforms),
  'val': datasets.ImageFolder(os.path.join(data, 'val'), transforms)
}

dataset_sizes = {x: len(img_datasets[x]) for x in ['train', 'val']}

dataloaders = {
  'train': torch.utils.data.DataLoader(img_datasets['train'], batch_size = 8),
  'val': torch.utils.data.DataLoader(img_datasets['val'], batch_size = 16)
}

class_names = img_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model, criterion, optimizer, scheduler, num_epochs = 20):

  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0

  for epoch in range(num_epochs):

    running_loss = 0.0
    running_corrects = 0.0
    print ('Epoch {}/{}'.format(epoch+1, num_epochs))

    for phase in ['train', 'val']:
      if phase == 'train':
        scheduler.step()
        model.train()
      else:
        model.eval()

      for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
          outputs = model(inputs)
          _, preds = torch.max(outputs, 1)
          loss = criterion(outputs, labels)

        running_loss += loss
        running_corrects += torch.sum(preds == labels.data)

      epoch_loss = running_loss / dataset_sizes[phase]
      epoch_acc = running_corrects.double() / dataset_sizes[phase]

      print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        phase, epoch_loss, epoch_acc))

      # deep copy the model
      if phase == 'val' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())

  print('Best val Acc: {:4f}'.format(best_acc))

  # load best model weights
  model.load_state_dict(best_model_wts)
  return model

def main():
  use_pretrained = False  # use pretrained model
  last_layer = False      # train last layer only

  try:
    opts, args = getopt.getopt(sys.argv[1:], 'pl', ['pretrained', 'last-layer'])
  except getopt.GetoptError as err:
    print(err)
    print('Usage: python ants_bees.py [--pretrained] [--last-layer]')
    sys.exit(2)
  
  for opt, arg in opts:
    if opt in ('-p', '--pretrained'):
      use_pretrained = True
    elif opt in ('-l', '--last-layer'):
      last_layer = True

  print("Running training with {}pretrained model and training {}.\n".format('' if use_pretrained else 'non-', 'last layer only' if last_layer else 'all layers'))
  
  model = models.resnet18(pretrained=use_pretrained)

  if last_layer:
    for param in model.parameters():
      param.requires_grad = False

  num_fts = model.fc.in_features
  model.fc = nn.Linear(num_fts, 2)
  model = model.to(device)

  criterion = nn.CrossEntropyLoss()

  # Observe that all parameters are being optimized
  params_to_train = model.fc.parameters() if last_layer else model.parameters()
  optimizer = optim.SGD(params_to_train, lr=0.001, momentum=0.9)

  # Decay LR by a factor of 0.1 every 7 epochs
  scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

  model = train(model, criterion, optimizer, scheduler, num_epochs=1)


if __name__ == '__main__':
  main()

