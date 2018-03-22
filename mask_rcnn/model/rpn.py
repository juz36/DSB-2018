"""
RPN definitions

"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from model.lib.bbox.overlap import bbox_overlaps_batch
from model.lib.bbox.transform import bbox_transform_batch, clip_boxes, clip_boxes_batch
from model.loss import smooth_l1_loss
from model.lib.bbox.generate_anchors import generate_anchors
from model.lib.bbox.nms import torch_nms as nms

# ---------------------------------------------------------------
# layers

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FC(nn.Module):
    def __init__(self, in_features, out_features, relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


# ---------------------------------------------------------------
# RPN

class RPN(nn.Module):
    def __init__(self, input_dims, config, training=True):
        super(RPN, self).__init__()
        self.config = config
        self.training = training
        self.anchor_scales = config.RPN_ANCHOR_SCALES
        self.anchor_ratios = config.RPN_ANCHOR_RATIOS
        # TBD: input dim, kernel size, padding, stride, output dim
        self.rpn_conv = Conv2d(input_dims, 512, 3, stride=1, same_padding=True)

        # fg/bg
        self.score_out = len(self.anchor_scales)*len(self.anchor_ratios)*2
        self.score_conv = Conv2d(512, self.score_out, 3, relu=False, same_padding=True)

        # box
        self.bbox_out = len(self.anchor_scales)*len(self.anchor_ratios)*4
        self.bbox_conv = Conv2d(512, self.bbox_out, 3, relu=False, same_padding=True)

        # proposal layer
        self.RPN_proposal = ProposalLayer(self.config)

        # anchor target layer
        self.RPN_anchor_target = AnchorTargetLayer(self.config)

        # loss
        self.rpn_cls_loss = 0
        self.rpn_bbox_loss = 0

    def forward(self, x, gt_boxes=None):
        batch_size = x.size(0)

        rpn_conv = self.rpn_conv(x)

        # rpn classification score
        rpn_cls_score = self.score_conv(rpn_conv)
        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)

        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.score_out)
        print(rpn_cls_prob.shape)
        # rpn boxes
        rpn_bbox_pred = self.bbox_conv(rpn_conv)

        # proposal layer
        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data))

        self.rpn_cls_loss = 0
        self.rpn_bbox_loss = 0

        if self.training:
            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes))

            # classification loss
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            rpn_label = rpn_data[0].view(batch_size, -1)

            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0, rpn_keep)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())
            self.rpn_cls_loss = F.cross_entropy(rpn_cls_score, rpn_label)
            fg_cnt = torch.sum(rpn_label.data.ne(0))

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]

            # bbox regression loss
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            self.rpn_bbox_loss = smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])

        return rois, self.rpn_cls_loss, self.rpn_bbox_loss

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1]*input_shape[2])/float(d)),
            input_shape[3]
        )
        return x

class ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self, config):
        super(ProposalLayer, self).__init__()

        self.config = config

        self._feat_stride = config.BACKBONE_STRIDES
        anchors = generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                           config.RPN_ANCHOR_RATIOS,
                                           config.BACKBONE_SHAPES,
                                           config.BACKBONE_STRIDES,
                                           config.RPN_ANCHOR_STRIDE)
        self._anchors = []
        for i in range(len(config.RPN_ANCHOR_SCALES)):
            self._anchors.append(generate_anchors(config.RPN_ANCHOR_SCALES[i],
                                                  config.RPN_ANCHOR_RATIOS,
                                                  config.BACKNONE_SHAPES[i],
                                                  config.BACKBONE_STRIDES[i],
                                                  config.RPN_ANCHOR_STRIDE))
        self._anchors = torch.from_numpy(self._anchors).float()
        self._num_anchors = []
        for i in range(len(config.RPN_ANCHOR_SCALES)):
            self._num_anchors.append(self._anchors[i].size(0))
        self._num_anchors = torch.from_numpy(self._num_anchors).int()

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        # top[0].reshape(1, 5)
        #
        # # scores blob: holds scores for R regions of interest
        # if len(top) > 1:
        #     top[1].reshape(1, 1, 1, 1)

    def forward(self, input, level):

        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)


        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs
        scores = input[0][:, self._num_anchors[level]:, :, :]
        bbox_deltas = input[1]

        pre_nms_topN  = self.config.PRE_NMS_ROIS_TRAINING
        post_nms_topN = self.config.POST_NMS_ROIS_TRAINING
        nms_thresh    = self.config.RPN_NMS_THRESHOLD
        min_size      = self.config.RPN_MIN_SIZE

        batch_size = bbox_deltas.size(0)

        feat_height, feat_width = scores.size(2), scores.size(3)
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(scores).float()

        A = self._num_anchors[level]
        K = shifts.size(0)

        self._anchors = self._anchors.type_as(scores)
        # anchors = self._anchors.view(1, A, 4) + shifts.view(1, K, 4).permute(1, 0, 2).contiguous()
        anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        anchors = anchors.view(1, K * A, 4).expand(batch_size, K * A, 4)

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:

        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous()
        bbox_deltas = bbox_deltas.view(batch_size, -1, 4)

        # Same story for the scores:
        scores = scores.permute(0, 2, 3, 1).contiguous()
        scores = scores.view(batch_size, -1)

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas, batch_size)

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, self.config.IMAGE_MAX_DIM, batch_size)
        # proposals = clip_boxes_batch(proposals, im_info, batch_size)

        # assign the score to 0 if it's non keep.
        # keep = self._filter_boxes(proposals, min_size * im_info[:, 2])

        # trim keep index to make it euqal over batch
        # keep_idx = torch.cat(tuple(keep_idx), 0)

        # scores_keep = scores.view(-1)[keep_idx].view(batch_size, trim_size)
        # proposals_keep = proposals.view(-1, 4)[keep_idx, :].contiguous().view(batch_size, trim_size, 4)

        # _, order = torch.sort(scores_keep, 1, True)

        scores_keep = scores
        proposals_keep = proposals
        _, order = torch.sort(scores_keep, 1, True)

        output = scores.new(batch_size, post_nms_topN, 5).zero_()
        for i in range(batch_size):
            # # 3. remove predicted boxes with either height or width < threshold
            # # (NOTE: convert min_size to input image scale stored in im_info[2])
            proposals_single = proposals_keep[i]
            scores_single = scores_keep[i]

            # # 4. sort all (proposal, score) pairs by score from highest to lowest
            # # 5. take top pre_nms_topN (e.g. 6000)
            order_single = order[i]

            if pre_nms_topN > 0 and pre_nms_topN < scores_keep.numel():
                order_single = order_single[:pre_nms_topN]

            proposals_single = proposals_single[order_single, :]
            scores_single = scores_single[order_single].view(-1,1)

            # 6. apply nms (e.g. threshold = 0.7)
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)

            keep_idx_i = nms(torch.cat((proposals_single, scores_single), 1), nms_thresh)
            keep_idx_i = keep_idx_i.long().view(-1)

            if post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:post_nms_topN]
            proposals_single = proposals_single[keep_idx_i, :]
            scores_single = scores_single[keep_idx_i, :]

            # padding 0 at the end.
            num_proposal = proposals_single.size(0)
            output[i,:,0] = i
            output[i,:num_proposal,1:] = proposals_single

        return output

    def _filter_boxes(self, boxes, min_size):
        """Remove all boxes with any side smaller than min_size."""
        ws = boxes[:, :, 2] - boxes[:, :, 0] + 1
        hs = boxes[:, :, 3] - boxes[:, :, 1] + 1
        keep = ((ws >= min_size.view(-1,1).expand_as(ws)) & (hs >= min_size.view(-1,1).expand_as(hs)))
        return keep


class AnchorTargetLayer(nn.Module):
    """
        Assign anchors to ground-truth targets. Produces anchor classification
        labels and bounding-box regression targets.
    """
    def __init__(self, config):
        super(AnchorTargetLayer, self).__init__()

        self.config = config

        self._feat_stride = config.BACKBONE_STRIDES
        self._scales = config.RPN_ANCHOR_SCALES
        anchors = generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                           config.RPN_ANCHOR_RATIOS,
                                           config.BACKBONE_SHAPES,
                                           config.BACKBONE_STRIDES,
                                           config.RPN_ANCHOR_STRIDE)
        self._anchors = torch.from_numpy(anchors).float()
        self._num_anchors = self._anchors.size(0)

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = 0  # default is 0

    def forward(self, input):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors

        rpn_cls_score = input[0]
        gt_boxes = input[1]

        # map of shape (..., H, W)
        height, width = rpn_cls_score.size(2), rpn_cls_score.size(3)

        batch_size = gt_boxes.size(0)

        feat_height, feat_width = rpn_cls_score.size(2), rpn_cls_score.size(3)
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(rpn_cls_score).float()

        A = self._num_anchors
        K = shifts.size(0)

        self._anchors = self._anchors.type_as(gt_boxes) # move to specific gpu.
        all_anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        all_anchors = all_anchors.view(K * A, 4)

        total_anchors = int(K * A)

        keep = ((all_anchors[:, 0] >= -self._allowed_border) &
                (all_anchors[:, 1] >= -self._allowed_border) &
                (all_anchors[:, 2] < long(self.config.IMAGE_MAX_DIM) + self._allowed_border) &
                (all_anchors[:, 3] < long(self.config.IMAGE_MAX_DIM) + self._allowed_border))

        inds_inside = torch.nonzero(keep).view(-1)

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = gt_boxes.new(batch_size, inds_inside.size(0)).fill_(-1)
        bbox_inside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()
        bbox_outside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()

        overlaps = bbox_overlaps_batch(anchors, gt_boxes)

        max_overlaps, argmax_overlaps = torch.max(overlaps, 2)
        gt_max_overlaps, _ = torch.max(overlaps, 1)

        if not self.config.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < self.config.RPN_NEGATIVE_OVERLAP] = 0

        gt_max_overlaps[gt_max_overlaps==0] = 1e-5
        keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size,1,-1).expand_as(overlaps)), 2)

        if torch.sum(keep) > 0:
            labels[keep>0] = 1

        # fg label: above threshold IOU
        labels[max_overlaps >= self.config.RPN_POSITIVE_OVERLAP] = 1

        if self.config.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < self.config.RPN_NEGATIVE_OVERLAP] = 0

        num_fg = int(self.config.ROI_POSITIVE_RATIO * self.config.TRAIN_ROIS_PER_IMAGE)

        sum_fg = torch.sum((labels == 1).int(), 1)
        sum_bg = torch.sum((labels == 0).int(), 1)

        for i in range(batch_size):
            # subsample positive labels if we have too many
            if sum_fg[i] > num_fg:
                fg_inds = torch.nonzero(labels[i] == 1).view(-1)
                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                #rand_num = torch.randperm(fg_inds.size(0)).type_as(gt_boxes).long()
                rand_num = torch.from_numpy(np.random.permutation(fg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = fg_inds[rand_num[:fg_inds.size(0)-num_fg]]
                labels[i][disable_inds] = -1

            num_bg = self.config.TRAIN_ROIS_PER_IMAGE - sum_fg[i]

            # subsample negative labels if we have too many
            if sum_bg[i] > num_bg:
                bg_inds = torch.nonzero(labels[i] == 0).view(-1)
                #rand_num = torch.randperm(bg_inds.size(0)).type_as(gt_boxes).long()

                rand_num = torch.from_numpy(np.random.permutation(bg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = bg_inds[rand_num[:bg_inds.size(0)-num_bg]]
                labels[i][disable_inds] = -1

        offset = torch.arange(0, batch_size)*gt_boxes.size(1)

        argmax_overlaps = argmax_overlaps + offset.view(batch_size, 1).type_as(argmax_overlaps)
        bbox_targets = _compute_targets_batch(anchors, gt_boxes.view(-1,5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5))

        # use a single value instead of 4 values for easy index.
        bbox_inside_weights[labels==1] = self.config.RPN_BBOX_INSIDE_WEIGHTS[0]

        if self.config.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            num_examples = torch.sum(labels[i] >= 0)
            positive_weights = 1.0 / num_examples
            negative_weights = 1.0 / num_examples
        else:
            assert ((self.config.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (self.config.TRAIN.RPN_POSITIVE_WEIGHT < 1))

        bbox_outside_weights[labels == 1] = positive_weights
        bbox_outside_weights[labels == 0] = negative_weights

        labels = _unmap(labels, total_anchors, inds_inside, batch_size, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, batch_size, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, batch_size, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, batch_size, fill=0)

        outputs = []

        labels = labels.view(batch_size, height, width, A).permute(0,3,1,2).contiguous()
        labels = labels.view(batch_size, 1, A * height, width)
        outputs.append(labels)

        bbox_targets = bbox_targets.view(batch_size, height, width, A*4).permute(0,3,1,2).contiguous()
        outputs.append(bbox_targets)

        anchors_count = bbox_inside_weights.size(1)
        bbox_inside_weights = bbox_inside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 4)

        bbox_inside_weights = bbox_inside_weights.contiguous().view(batch_size, height, width, 4*A)\
                            .permute(0,3,1,2).contiguous()

        outputs.append(bbox_inside_weights)

        bbox_outside_weights = bbox_outside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 4)
        bbox_outside_weights = bbox_outside_weights.contiguous().view(batch_size, height, width, 4*A)\
                            .permute(0,3,1,2).contiguous()
        outputs.append(bbox_outside_weights)

        return outputs

def _unmap(data, count, inds, batch_size, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """

    if data.dim() == 2:
        ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
        ret[:, inds] = data
    else:
        ret = torch.Tensor(batch_size, count, data.size(2)).fill_(fill).type_as(data)
        ret[:, inds,:] = data
    return ret


def _compute_targets_batch(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    return bbox_transform_batch(ex_rois, gt_rois[:, :, :4])
