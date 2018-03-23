"""
Training mask r-cnn

"""
from utils.setup import *
from utils.path import train_dir, model_dir, result_dir, log_dir
from model.mask_rcnn import MaskRCNN
from dataset.dataset import CellDataset, train_collate
from model.loss import total_loss

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.optim import Adam, SGD

import time
from tqdm import trange
from config import Config
from utils.logger import TFLogger, Statistics, StatCollection


# configurations
lr = Config.LEARNING_RATE
mom = Config.LEARNING_MOMENTUM
num_epoch = 10
batch_size = Config.IMAGES_PER_GPU
print_freq = 10

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
    num_workers=4,
    collate_fn=train_collate)

# model setup
model = MaskRCNN(config).cuda()

#optimizer = Adam(model.parameters(), lr=lr)
optimizer = SGD(model.parameters(), lr=lr, momentum=mom)
tensorboard = TFLogger(log_dir)

log_list = [
    'rpn_cls_loss',
    'rpn_reg_loss',
    'stage2_cls_loss',
    'stage2_reg_loss',
    'stage2_mask_loss',
    'total_loss']
collection = StatCollection(log_list)


step_global = 0
for epoch in range(num_epoch):
    with trange(len(train_loader), desc="Epoch %d" % epoch) as pbar:
        for i, (imgs, gts) in enumerate(train_loader):
            imgs = imgs.float().cuda()

            # compute loss
            logits = model.forward(imgs)
            loss, saved_for_log = total_loss(logits, gts, config)

            # learn & update params
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            collection.update_dict(saved_for_log)
            
            # print logs
            if i % print_freq == print_freq-1:
                pbar.set_postfix_str(
                    'Loss: Total {total_loss:.4f}    '
                    'RPN Cls {rpn_cls_loss:.4f}    '
                    'RPN Bbox {rpn_reg_loss:.4f}    '
                    'Rcnn Cls {stage2_cls_loss:.4f}    '
                    'Rcnn Bbox {stage2_reg_loss:.4f}    '
                    'Mask {stage2_mask_loss:.4f}'.format(**saved_for_log))

                collection.summary_all(tensorboard, step_global)
                collection.reset_all()

            pbar.update()
            step_global += 1
