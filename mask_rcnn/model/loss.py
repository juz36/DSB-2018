import torch
import torch.nn.functional as F
import numpy as np

################
#Loss functions#
################
# region proposal network confidence loss
def rpn_class_loss(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = torch.eq(rpn_match, 1)
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = torch.ne(rpn_match, 0.)

    rpn_class_logits = torch.masked_select(rpn_class_logits, indices)
    anchor_class = torch.masked_select(anchor_class, indices)

    rpn_class_logits = rpn_class_logits.contiguous().view(-1, 2)

    anchor_class = anchor_class.contiguous().view(-1).type(torch.cuda.LongTensor)
    loss = F.cross_entropy(rpn_class_logits, anchor_class, weight=None)
    return loss

# region proposal bounding bbox loss
def rpn_bbox_loss(target_bbox, rpn_match, rpn_bbox, config):
    """Return the RPN bounding box loss graph.
    config: the model config object.
    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """
    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    indices = torch.eq(rpn_match, 1)
    rpn_bbox = torch.masked_select(rpn_bbox, indices)
    batch_counts = torch.sum(indices.float(), dim=1)

    outputs = []
    for i in range(config.IMAGES_PER_GPU):
#        print(batch_counts[i].cpu().data.numpy()[0])
        outputs.append(target_bbox[torch.cuda.LongTensor([i]), torch.arange(int(batch_counts[i].cpu().data.numpy()[0])).type(torch.cuda.LongTensor)])

    target_bbox = torch.cat(outputs, dim=0)

    loss = F.smooth_l1_loss(rpn_bbox, target_bbox, size_average=True)
    return loss

# rcnn head confidence loss
def rcnn_class_loss(target_class_ids, pred_class_logits, config):
    """Loss for the classifier head of Mask RCNN.
    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    """

    # Find predictions of classes that are not in the dataset.
    pred_class_logits = pred_class_logits.contiguous().view(-1, config.NUM_CLASSES)

    target_class_ids = target_class_ids.contiguous().view(-1).type(torch.cuda.LongTensor)
    # Loss
    loss = F.cross_entropy(
        pred_class_logits, target_class_ids, weight=None, size_average=True)

    # Erase losses of predictions of classes that are not in the active
    # classes of the image.
#    loss = loss * pred_active

    # Computer loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.
#    loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
    return loss

# rcnn head bbox loss
def rcnn_bbox_loss(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.
    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """
    # Reshape to merge batch and roi dimensions for simplicity.
    target_class_ids = target_class_ids.contiguous().view(-1)
    target_bbox = target_bbox.contiguous().view(-1, 4)
    pred_bbox = pred_bbox.contiguous().view(-1, pred_bbox.size()[2], 4)
#    print(target_class_ids)

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indicies.
    positive_roi_ix = torch.gt(target_class_ids , 0)
#    print(positive_roi_ix)
    positive_roi_class_ids = torch.masked_select(target_class_ids, positive_roi_ix)

    indices = target_class_ids
#    indices = torch.stack([positive_roi_ix, positive_roi_class_ids], dim=1)
#    print(indices)
    # Gather the deltas (predicted and true) that contribute to loss
#    target_bbox = torch.gather(target_bbox, positive_roi_ix)
#    pred_bbox = torch.gather(pred_bbox, indices)

    loss = F.smooth_l1_loss(pred_bbox, target_bbox, size_average=True)
    return loss

# rcnn head mask loss
def mask_loss(target_masks, target_class_ids, pred_masks_logits):
    """Mask binary cross-entropy loss for the masks head.
    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = target_class_ids.view(-1)

    loss = F.binary_cross_entropy_with_logits(pred_masks_logits, target_masks)
    return loss

# total loss
def total_loss(saved_for_loss, ground_truths, config):
    #create dict to save loss for visualization
    saved_for_log = OrderedDict()
    #unpack saved log
    predict_rpn_class_logits, predict_rpn_class,\
    predict_rpn_bbox, predict_rpn_rois,\
    predict_mrcnn_class_logits, predict_mrcnn_class,\
    predict_mrcnn_bbox, predict_mrcnn_masks_logits = saved_for_loss

    batch_rpn_match, batch_rpn_bbox, \
    batch_gt_class_ids, batch_gt_boxes,\
    batch_gt_masks = ground_truths


    rpn_rois = predict_rpn_rois.cpu().data.numpy()
    rpn_rois = rpn_rois[:, :, [1, 0, 3, 2]]
    batch_rois, batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask = stage2_target(rpn_rois, batch_gt_class_ids, batch_gt_boxes, batch_gt_masks, config)

#        print(np.sum(batch_mrcnn_class_ids))
    batch_mrcnn_mask = batch_mrcnn_mask.transpose(0, 1, 4, 2, 3)
    batch_mrcnn_class_ids = to_variable(
        batch_mrcnn_class_ids).cuda()
    batch_mrcnn_bbox = to_variable(batch_mrcnn_bbox).cuda()
    batch_mrcnn_mask = to_variable(batch_mrcnn_mask).cuda()

#        print(batch_mrcnn_class_ids)
    # RPN branch loss->classification
    rpn_cls_loss = rpn_class_loss(
        batch_rpn_match, predict_rpn_class_logits)

    # RPN branch loss->bbox
    rpn_reg_loss = rpn_bbox_loss(
        batch_rpn_bbox, batch_rpn_match, predict_rpn_bbox, config)

    # bbox branch loss->bbox
    stage2_reg_loss = rcnn_bbox_loss(
        batch_mrcnn_bbox, batch_mrcnn_class_ids, predict_mrcnn_bbox)

    # cls branch loss->classification
    stage2_cls_loss = rcnn_class_loss(
        batch_mrcnn_class_ids, predict_mrcnn_class_logits, config)

    # mask branch loss
    stage2_mask_loss = mask_loss(
        batch_mrcnn_mask, batch_mrcnn_class_ids, predict_mrcnn_masks_logits)

    total_loss = rpn_cls_loss + rpn_reg_loss + stage2_cls_loss + stage2_reg_loss + stage2_mask_loss
    saved_for_log['rpn_cls_loss'] = rpn_cls_loss.data[0]
    saved_for_log['rpn_reg_loss'] = rpn_reg_loss.data[0]
    saved_for_log['stage2_cls_loss'] = stage2_cls_loss.data[0]
    saved_for_log['stage2_reg_loss'] = stage2_reg_loss.data[0]
    saved_for_log['stage2_mask_loss'] = stage2_mask_loss.data[0]
    saved_for_log['total_loss'] = total_loss.data[0]

    return total_loss, saved_for_log

def stage2_target(rpn_rois, gt_class_ids, gt_boxes, gt_masks, config):

    batch_rois = []
    batch_mrcnn_class_ids = []
    batch_mrcnn_bbox = []
    batch_mrcnn_mask = []

    for i in range(config.IMAGES_PER_GPU):
        rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask =\
        build_detection_targets(
        rpn_rois[i], gt_class_ids[i], gt_boxes[i], gt_masks[i], config)

        batch_rois.append(rois)
        batch_mrcnn_class_ids.append(mrcnn_class_ids)
        batch_mrcnn_bbox.append(mrcnn_bbox)
        batch_mrcnn_mask.append(mrcnn_mask)

    batch_rois = np.array(batch_rois)
    batch_mrcnn_class_ids = np.array(batch_mrcnn_class_ids)
    batch_mrcnn_bbox = np.array(batch_mrcnn_bbox)
    batch_mrcnn_mask = np.array(batch_mrcnn_mask)
    return batch_rois, batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask
