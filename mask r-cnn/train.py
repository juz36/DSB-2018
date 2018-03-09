"""
Training mask r-cnn

"""
from utils.setup import *
from utils.path import train_dir, model_dir, result_dir, log_dir
from model.resnet import MaskRCnn
from dataset.dataset import CellDataset

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim import Adam
import torch.nn as nn


# configurations
lr = 1e-3
num_epoch = 100
batch_size = 8

# data transformations
transform = transforms.Compose([
    transforms.Resize(),
    transforms.ToTensor(),
    transforms.Normalize()
])

# load data
train_dataset = CellDataset(train_dir, transform=transform)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4)

# model setup
model = MaskRCnn.cuda()
loss_function = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=lr)

for epoch in range(num_epoch):
    running_loss = 0
    for i, data in enumerate(train_loader):
        imgs, masks = data
        imgs = Variable(imgs)
        masks = Variable(masks)

        # compute loss
        logits = model(imgs)
        loss = loss_function(logits, masks)

        # learn & update params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print logs
        running_loss += loss.data[0]

        if i % 500 == 499:
            print("loss: ", running_loss/500)
            running_loss = 0

