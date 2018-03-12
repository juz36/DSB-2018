"""
Model definitions

"""
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

from model.resnet import resnet50
from model.rpn import RPN
from model.lib.roi_align.roi_align.roi_align import RoIAlign


# ---------------------------------------------------------------
# Heads

class MaskHead(nn.Module):

    def __init__(self, cfg, in_channels):
        super(MaskHead, self).__init__()
        self.num_classes = cfg.num_classes
        self.crop_size = cfg.mask_crop_size

        self.roi_align = RoIAlign(self.crop_size, self.crop_size)
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=4, padding=1, stride=2, bias=False)
        self.mask = nn.Conv2d(256, self.num_classes, kernel_size=1, padding=0, stride=1)

    def forward(self, x, rpn_rois):
        x = self.roi_align(x, rpn_rois)

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = F.relu(self.bn4(self.conv4(x)), inplace=True)
        x = self.deconv(x)
        rcnn_mask_logits = self.mask(x)

        return rcnn_mask_logits


class RCNNHead(nn.Module):
    def __init__(self, cfg, in_channels):
        super(RcnnHead, self).__init__()
        self.num_classes = cfg.num_classes
        self.crop_size = cfg.rcnn_crop_size

        self.roi_align = RoIAlign(self.crop_size, self.crop_size)
        self.fc1 = nn.Linear(in_channels * self.crop_size * self.crop_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.class_logits = nn.Linear(1024, self.num_classes)
        self.bbox = nn.Linear(1024, self.num_classes * 4)

    def forward(self, x, rpn_rois):
        x = self.roi_align(x, rpn_rois)
        #x = crops.view(crops.size(0), -1)
        #x = F.relu(self.fc1(x), inplace=True)
        #x = F.relu(self.fc2(x), inplace=True)
        #x = F.dropout(x, 0.5, training=self.training)
        rcnn_class_logits = self.class_logits(x)
        rcnn_bbox = self.bbox(x)

        return rcnn_class_logits, rcnn_bbox


class Crop(nn.Module):
    def __init__(self, cfg):
        super(Crop, self).__init__()
        self.num_scales = len(cfg.rpn_scales)
        self.crop_size = cfg.crop_size
        self.sizes = cfg.rpn_base_sizes
        self.scales = cfg.rpn_scales

        #self.crops = nn.ModuleList()
        #for l in range(self.num_scales):
        #    self.crops.append(
        #        Crop(self.crop_size, self.crop_size, 1 / self.scales[l])
        #    )
        #self.crop = RoIAlign()

    def forward(self, fs, proposals):
        num_proposals = len(proposals)

        ## this is  complicated. we need to decide for a given roi, which of the p0,p1, ..p3 layers to pool from
        boxes = proposals.detach().data[:, 1:5]
        sizes = boxes[:, 2:] - boxes[:, :2]
        sizes = torch.sqrt(sizes[:, 0] * sizes[:, 1])
        distances = torch.abs(sizes.view(num_proposals, 1).expand(num_proposals, 4) \
                              - torch.from_numpy(np.array(self.sizes, np.float32)).cuda())
        min_distances, min_index = distances.min(1)

        rois = proposals.detach().data[:, 0:5]
        rois = Variable(rois)

        crops = []
        indices = []
        for l in range(self.num_scales):
            index = (min_index == l).nonzero()

            if len(index) > 0:
                crop = self.crops[l](fs[l], rois[index].view(-1, 5))
                crops.append(crop)
                indices.append(index)

        crops = torch.cat(crops, 0)
        indices = torch.cat(indices, 0).view(-1)
        crops = crops[torch.sort(indices)[1]]
        # crops = torch.index_select(crops,0,index)

        return crops


# ---------------------------------------------------------------
# Mask R-CNN

class MaskRCNN(nn.Module):

    def __init__(self, cfg):
        super(MaskRCnn, self).__init__()
        self.__mode = 'train'
        feature_channels = 128
        # define modules (set of layers)
#        self.feature_net = FeatureNet(cfg, 3, feature_channels)
        self.feature_net = resnet50()
        #self.rpn_head = RpnMultiHead(cfg,feature_channels)
        self.rpn = RPN(cfg, feature_channels)
        #self.rcnn_crop = CropRoi(cfg, cfg.rcnn_crop_size)
        self.rcnn_head = RCNNHead(cfg, crop_channels)
        #self.mask_crop = CropRoi(cfg, cfg.mask_crop_size)
        self.mask_head = MaskHead(cfg, crop_channels)



    def forward(self, x):
        # Extract features
        features = self.feature_net(x)
        #print(features.shape)

        # RPN proposals
        rpn_class_logits, rpn_bbox, rpn_proposals = self.rpn(features)
        #print(rpn_proposals.shape)
        #rois = rpn_nms()

        #if mode in ['train', 'validation']:


        # RCNN proposals
        rcnn_class_logits, rcnn_bbox = self.rcnn_head(features, rpn_proposals)

        rcnn_mask_logits = self.mask_head(features, rpn_proposals)


        pass
