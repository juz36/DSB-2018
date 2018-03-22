"""
Model definitions

"""
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math

from model.resnet import resnet50
from model.rpn import RPN
from model.proposal_target_layer import ProposalTargetLayer
from model.fpn import FPN
#from model.lib.roi_align.roi_align.roi_align import RoIAlign
from model.lib.roi_align.roi_align.crop_and_resize import CropAndResize
from model.lib.bbox.generate_anchors import generate_pyramid_anchors
from model.lib.bbox.nms import torch_nms as nms


def log2_graph(x):
    """Implementatin of Log2. pytorch doesn't have a native implemenation."""
    return torch.div(torch.log(x), math.log(2.))


def ROIAlign(feature_maps, rois, config, pool_size, mode='bilinear'):
    """Implements ROI Align on the features.
    Params:
    - pool_shape: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, chanells]. Shape of input image in pixels
    Inputs:
    - boxes: [batch, num_boxes, (x1, y1, x2, y2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, channels, height, width]
    Output:
    Pooled regions in the shape: [batch, num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """
    """
    [  x2-x1             x1 + x2 - W + 1  ]
    [  -----      0      ---------------  ]
    [  W - 1                  W - 1       ]
    [                                     ]
    [           y2-y1    y1 + y2 - H + 1  ]
    [    0      -----    ---------------  ]
    [           H - 1         H - 1      ]
    """
    #feature_maps= [P2, P3, P4, P5]
    rois = rois.detach()
    crop_resize = CropAndResize(pool_size, pool_size, 0)

    roi_number = rois.size()[1]

    pooled = rois.data.new(
            config.IMAGES_PER_GPU*rois.size(
            1), 256, pool_size, pool_size).zero_()

    rois = rois.view(
            config.IMAGES_PER_GPU*rois.size(1),
            4)

    # Loop through levels and apply ROI pooling to each. P2 to P5.
    x_1 = rois[:, 0]
    y_1 = rois[:, 1]
    x_2 = rois[:, 2]
    y_2 = rois[:, 3]

    roi_level = log2_graph(
        torch.div(torch.sqrt((y_2 - y_1) * (x_2 - x_1)), 224.0))

    roi_level = torch.clamp(torch.clamp(
        torch.add(torch.round(roi_level), 4), min=2), max=5)

    # P2 is 256x256, P3 is 128x128, P4 is 64x64, P5 is 32x32
    # P2 is 4, P3 is 8, P4 is 16, P5 is 32
    for i, level in enumerate(range(2, 6)):

        scaling_ratio = 2**level

        height = float(config.IMAGE_MAX_DIM)/ scaling_ratio
        width = float(config.IMAGE_MAX_DIM) / scaling_ratio

        ixx = torch.eq(roi_level, level)

        box_indices = ixx.view(-1).int() * 0
        ix = torch.unsqueeze(ixx, 1)
        level_boxes = torch.masked_select(rois, ix)
        if level_boxes.size()[0] == 0:
            continue
        level_boxes = level_boxes.view(-1, 4)

        crops = crop_resize(feature_maps[i], torch.div(
                level_boxes, float(config.IMAGE_MAX_DIM)
                )[:, [1, 0, 3, 2]], box_indices)

        indices_pooled = ixx.nonzero()[:, 0]
        pooled[indices_pooled.data, :, :, :] = crops.data

    pooled = pooled.view(config.IMAGES_PER_GPU, roi_number,
               256, pool_size, pool_size)
    pooled = Variable(pooled).cuda()
    return pooled


# ---------------------------------------------------------------
# Heads

class MaskHead(nn.Module):

    def __init__(self, config):
        super(MaskHead, self).__init__()
        self.config = config
        self.num_classes = config.NUM_CLASSES
        #self.crop_size = config.mask_crop_size

        #self.roi_align = RoIAlign(self.crop_size, self.crop_size)
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
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
        #x = self.roi_align(x, rpn_rois)
        x = ROIAlign(x, rpn_rois, self.config, self.config.MASK_POOL_SIZE)

        roi_number = x.size()[1]

        # merge batch and roi number together
        x = x.view(self.config.IMAGES_PER_GPU * roi_number,
                   256, self.config.MASK_POOL_SIZE,
                   self.config.MASK_POOL_SIZE)

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = F.relu(self.bn4(self.conv4(x)), inplace=True)
        x = self.deconv(x)
        rcnn_mask_logits = self.mask(x)

        rcnn_mask_logits = rcnn_mask_logits.view(self.config.IMAGES_PER_GPU,
                                                 roi_number,
                                                 self.config.NUM_CLASSES,
                                                 self.config.MASK_POOL_SIZE * 2,
                                                 self.config.MASK_POOL_SIZE * 2)

        return rcnn_mask_logits


class RCNNHead(nn.Module):
    def __init__(self, config):
        super(RCNNHead, self).__init__()
        self.config = config
        self.num_classes = config.NUM_CLASSES
        #self.crop_size = config.rcnn_crop_size

        #self.roi_align = RoIAlign(self.crop_size, self.crop_size)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.class_logits = nn.Linear(1024, self.num_classes)
        self.bbox = nn.Linear(1024, self.num_classes * 4)

        self.conv1 = nn.Conv2d(256, 1024, kernel_size=self.config.POOL_SIZE, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(1024, eps=0.001)

    def forward(self, x, rpn_rois):
        x = ROIAlign(x, rpn_rois, self.config, self.config.POOL_SIZE)
        roi_number = x.size()[1]

        x = x.view(self.config.IMAGES_PER_GPU * roi_number,
                   256, self.config.POOL_SIZE,
                   self.config.POOL_SIZE)
        #print(x.shape)
        #x = self.roi_align(x, rpn_rois, self.config, self.config.POOL_SIZE)
        #x = crops.view(crops.size(0), -1)
        x = self.bn1(self.conv1(x))
        x = x.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        #x = F.dropout(x, 0.5, training=self.training)
        rcnn_class_logits = self.class_logits(x)
        rcnn_probs = F.softmax(rcnn_class_logits, dim=-1)

        rcnn_bbox = self.bbox(x)

        rcnn_class_logits = rcnn_class_logits.view(self.config.IMAGES_PER_GPU,
                                                   roi_number,
                                                   rcnn_class_logits.size()[-1])

        rcnn_probs = rcnn_probs.view(self.config.IMAGES_PER_GPU,
                                     roi_number,
                                     rcnn_probs.size()[-1])

        rcnn_bbox = rcnn_bbox.view(self.config.IMAGES_PER_GPU,
                                   roi_number,
                                   self.config.NUM_CLASSES,
                                   4)

        return rcnn_class_logits, rcnn_probs, rcnn_bbox


#
# ---------------------------------------------------------------
# Mask R-CNN

class MaskRCNN(nn.Module):
    """
    Mask R-CNN model
    """

    def __init__(self, config, training=True):
        super(MaskRCNN, self).__init__()
        self.config = config
        self.training = training
        feature_channels = 128
        # define modules (set of layers)
#        self.feature_net = FeatureNet(cfg, 3, feature_channels)
        self.feature_net = resnet50().cuda()

        # FPN
        self.fpn = FPN()

        # RPN
        self.rpn = RPN(256, self.config)
        self.rpn_proposal_target = ProposalTargetLayer(self.config)

        self.rcnn_head = RCNNHead(config)
        self.mask_head = MaskHead(config)

        self.proposal_count = self.config.POST_NMS_ROIS_TRAINING

        self.scale_ratios = [4, 8, 16, 32]
        # loss
        self.rcnn_cls_loss = 0
        self.rcnn_bbox_loss = 0
        self.mask_loss = 0

    def forward(self, x, gt_boxes, gt_masks):
        # Extract features
        C1, C2, C3, C4, C5 = self.feature_net(x)
        P2, P3, P4, P5, P6 = self.fpn(C2, C3, C4, C5)

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2, P3, P4, P5, P6]

        self.mrcnn_feature_maps = [P2, P3, P4, P5]

        rpn_class_logits_outputs = []
        rpn_class_outputs = []
        rpn_bbox_outputs = []
        # RPN proposals
        for i in range(len(rpn_feature_maps)):
            print(self.config.BACKBONE_SHAPES[i])
            print(rpn_feature_maps[i].shape)
        for feature in rpn_feature_maps:
            rois, rpn_cls_loss, rpn_bbox_loss = self.rpn(feature, gt_boxes)
            rpn_rois_outputs.append(rois)
            rpn_cls_outputs.append(rpn_cls_loss)
            rpn_bbox_outputs.append(rpn_bbox_loss)

        rois = torch.cat(rpn_rois_outputs, dim=1)
        rpn_cls_loss = torch.cat(rpn_cls_outputs, dim=1)
        rpn_bbox_loss = torch.cat(rpn_bbox_outputs, dim=1)

        if self.training:
            roi_data = self.rpn_proposal_target(rois, gt_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)

        # RCNN proposals
        rcnn_class_logits, rcnn_class, rcnn_bbox = self.rcnn_head(self.mrcnn_feature_maps, rois)
        rcnn_mask_logits = self.mask_head(self.mrcnn_feature_maps, rois)
        # <todo> mask nms

        return [rpn_cls_loss, rpn_bbox_loss, rois,
                rcnn_class_logits, rcnn_class, rcnn_bbox,
                rcnn_mask_logits]
