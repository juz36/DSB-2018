"""
Training mask r-cnn

"""
from utils.setup import *
from utils.path import train_dir, model_dir, result_dir, log_dir
from model.mask_rcnn import MaskRCNN
from dataset.dataset import CellDataset
from model.loss import total_loss

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim import Adam
import torch.nn as nn
import torch
from config import Config

# configurations
lr = 1e-3
num_epoch = 100
batch_size = 1

# data transformations
#transform = transforms.Compose([
#    transforms.Resize(),
#    transforms.ToTensor(),
#    transforms.Normalize()
#])

config = Config()
# load data
train_dataset = CellDataset(train_dir, config)#, transform=transform)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=1)

# model setup
model = MaskRCNN(config).cuda()
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
#loss_function = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=lr)

for epoch in range(num_epoch):
    running_loss = 0
    for batch_data in train_loader:
        imgs, gts = batch_data
        imgs = imgs.float().cuda()
        #print(imgs)

        # compute loss
        logits = model.forward(imgs)
        #gts = []
        loss, saved_for_log = total_loss(logits, gts, config)

        #loss = loss_function(logits, ground_truths)

        # learn & update params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print logs
        running_loss += saved_for_log.data[0]

        #if i % 500 == 499:
        #    print("loss: ", running_loss/500)
        #    running_loss = 0
