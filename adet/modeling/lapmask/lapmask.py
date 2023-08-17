"""Lapmask"""
# -*- coding: utf-8 -*-
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from detectron2.layers import ShapeSpec, batched_nms, cat, paste_masks_in_image
from detectron2.layers import batched_nms
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.structures.masks import PolygonMasks, polygons_to_bitmask
from detectron2.utils.logger import log_first_n
from fvcore.nn import sigmoid_focal_loss_jit
from torch import nn
from typing import List

from adet.utils.comm import aligned_bilinear
from .loss import dice_loss, FocalLoss, iou_loss
from .utils import Conv, ConvBlock, SPP, Dilated_bottleNeck
from .utils import imrescale, center_of_mass, point_nms, mask_nms, matrix_nms, DynamicDeformConv

__all__ = ["LapMask"]


@META_ARCH_REGISTRY.register()
class LapMask(nn.Module):
    """
    LapMask model. Creates FPN backbone, instance branch for kernels for each pyramid level and categories prediction,
    mask branch for unified mask features.
    Calculates and applies proper losses to class and masks.
    """

    def __init__(self, cfg):
        super().__init__()

        # get the device of the model
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.scale_ranges = cfg.MODEL.LAPMASK.FPN_SCALE_RANGES
        self.strides = cfg.MODEL.LAPMASK.FPN_INSTANCE_STRIDES
        self.sigma = cfg.MODEL.LAPMASK.SIGMA
        # Instance parameters.
        self.num_classes = cfg.MODEL.LAPMASK.NUM_CLASSES
        self.num_kernels = cfg.MODEL.LAPMASK.NUM_KERNELS
        self.num_grids = cfg.MODEL.LAPMASK.NUM_GRIDS

        self.instance_in_features = cfg.MODEL.LAPMASK.INSTANCE_IN_FEATURES
        self.instance_strides = cfg.MODEL.LAPMASK.FPN_INSTANCE_STRIDES
        self.instance_in_channels = cfg.MODEL.LAPMASK.INSTANCE_IN_CHANNELS  # = fpn.
        self.instance_channels = cfg.MODEL.LAPMASK.INSTANCE_CHANNELS

        # Mask parameters.
        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_in_features = cfg.MODEL.LAPMASK.MASK_IN_FEATURES
        self.mask_in_channels = cfg.MODEL.LAPMASK.MASK_IN_CHANNELS
        self.mask_channels = cfg.MODEL.LAPMASK.MASK_CHANNELS
        self.num_masks = cfg.MODEL.LAPMASK.NUM_MASKS

        self.mask_out_stride = 4  # use 4-stride gt mask.

        # Inference parameters.
        self.max_before_nms = cfg.MODEL.LAPMASK.NMS_PRE
        self.score_threshold = cfg.MODEL.LAPMASK.SCORE_THR
        self.update_threshold = cfg.MODEL.LAPMASK.UPDATE_THR
        self.maskness_threshold = cfg.MODEL.LAPMASK.MASKNESS_THR
        self.mask_threshold = cfg.MODEL.LAPMASK.MASK_THR
        self.max_per_img = cfg.MODEL.LAPMASK.MAX_PER_IMG
        self.nms_kernel = cfg.MODEL.LAPMASK.NMS_KERNEL
        self.nms_sigma = cfg.MODEL.LAPMASK.NMS_SIGMA
        self.nms_type = cfg.MODEL.LAPMASK.NMS_TYPE

        # build the backbone.
        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()

        # build the ins head.
        instance_shapes = [backbone_shape[f] for f in self.instance_in_features]
        self.ins_head = LapMaskInsHead(cfg, instance_shapes)

        # build the mask head.
        mask_shapes = [backbone_shape[f] for f in self.mask_in_features]
        self.mask_head = LapMaskHead(cfg, mask_shapes)

        # loss
        self.ins_loss_weight = cfg.MODEL.LAPMASK.LOSS.DICE_WEIGHT
        self.focal_loss_alpha = cfg.MODEL.LAPMASK.LOSS.FOCAL_ALPHA
        self.focal_loss_gamma = cfg.MODEL.LAPMASK.LOSS.FOCAL_GAMMA
        self.focal_loss_weight = cfg.MODEL.LAPMASK.LOSS.FOCAL_WEIGHT

        # image transform
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

        self.cnt = 0

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DetectionTransform` .
                Each item in the list contains the inputs for one image.
            For now, each item in the list is a dict that contains:
                image: Tensor, image in (C, H, W) format.
                instances: Instances
                Other information that's included in the original dicts, such as:
                    "height", "width" (int): the output resolution of the model, used in inference.
                        See :meth:`postprocess` for details.
         Returns:
            losses (dict[str: Tensor]): mapping from a named loss to a tensor
                storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            # self.add_bitmasks(gt_instances, images.tensor.size(-2), images.tensor.size(-1))
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]

        else:
            gt_instances = None

        if self.training:
            """
            get_ground_truth.
            return loss and so on.
            """
            features = self.backbone(images.tensor)

            # ins branch
            ins_features = [features[f] for f in self.instance_in_features]
            ins_features = self.split_feats(ins_features)
            cate_pred, kernel_pred = self.ins_head(ins_features)

            # mask branch
            mask_features = [features[f] for f in self.mask_in_features]
            mask_pred, offset_mask_pred = self.mask_head(mask_features)

            mask_feat_size = mask_pred[0].size()[-2:]
            targets = self.get_ground_truth(gt_instances, mask_feat_size, mask_pred[0].device)
            losses = self.loss(cate_pred, kernel_pred, mask_pred, offset_mask_pred, targets)
            return losses
        else:
            with torch.no_grad():
                features = self.backbone(images.tensor)

                # ins branch
                ins_features = [features[f] for f in self.instance_in_features]
                ins_features = self.split_feats(ins_features)
                cate_pred, kernel_pred = self.ins_head(ins_features)

                # mask branch
                mask_features = [features[f] for f in self.mask_in_features]
                mask_pred, offset_mask_pred = self.mask_head(mask_features)

                # point nms.
                cate_pred = [point_nms(cate_p.sigmoid(), kernel=2).permute(0, 2, 3, 1)
                             for cate_p in cate_pred]
                # do inference for results.
                results = self.inference(cate_pred, kernel_pred, mask_pred, offset_mask_pred, images.image_sizes,
                                         batched_inputs)
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @torch.no_grad()
    def get_ground_truth(self, gt_instances, mask_feat_size=None, device=None):
        """get_ground_truth"""
        ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list = [], [], [], []
        for img_idx in range(len(gt_instances)):
            cur_ins_label_list, cur_cate_label_list, \
            cur_ins_ind_label_list, cur_grid_order_list = \
                self.get_ground_truth_single(img_idx, gt_instances,
                                             mask_feat_size=mask_feat_size, device=device)
            ins_label_list.append(cur_ins_label_list)
            cate_label_list.append(cur_cate_label_list)
            ins_ind_label_list.append(cur_ins_ind_label_list)
            grid_order_list.append(cur_grid_order_list)
        return ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list

    def get_ground_truth_single(self, img_idx, gt_instances, mask_feat_size, device):
        """get_ground_single"""
        gt_bboxes_raw = gt_instances[img_idx].gt_boxes.tensor
        gt_labels_raw = gt_instances[img_idx].gt_classes
        gt_masks_raw = gt_instances[img_idx].gt_masks.tensor
        # ins
        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (
                gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))

        ins_label_list = []
        cate_label_list = []
        ins_ind_label_list = []
        grid_order_list = []
        for (lower_bound, upper_bound), stride, num_grid \
                in zip(self.scale_ranges, self.strides, self.num_grids):

            hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero(as_tuple=False).flatten()
            num_ins = len(hit_indices)

            ins_label = []
            grid_order = []
            cate_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device)
            cate_label = torch.fill_(cate_label, self.num_classes)
            ins_ind_label = torch.zeros([num_grid ** 2], dtype=torch.bool, device=device)

            if num_ins == 0:
                ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label)
                ins_ind_label_list.append(ins_ind_label)
                grid_order_list.append([])
                continue
            gt_bboxes = gt_bboxes_raw[hit_indices]
            gt_labels = gt_labels_raw[hit_indices]
            gt_masks = gt_masks_raw[hit_indices, ...]

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

            # mass center
            center_ws, center_hs = center_of_mass(gt_masks)
            valid_mask_flags = gt_masks.sum(dim=-1).sum(dim=-1) > 0

            output_stride = 4
            im_h, im_w = mask_feat_size
            h, w = gt_masks.shape[1:]
            gt_masks = F.pad(gt_masks, (0, im_w * 4 - w, 0, im_h * 4 - h), "constant", 0)
            gt_masks = F.interpolate(gt_masks.float().unsqueeze(0), scale_factor=0.25, mode='bilinear',
                                     align_corners=False).squeeze(0).bool()
            gt_masks = gt_masks.to(dtype=torch.uint8, device=device)

            for seg_mask, gt_label, half_h, half_w, center_h, center_w, valid_mask_flag in zip(gt_masks, gt_labels,
                                                                                               half_hs, half_ws,
                                                                                               center_hs, center_ws,
                                                                                               valid_mask_flags):
                if not valid_mask_flag:
                    continue
                upsampled_size = (mask_feat_size[0] * 4, mask_feat_size[1] * 4)
                # coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))
                # coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))
                coord_w = int(torch.div(center_w / upsampled_size[1], 1. / num_grid))
                coord_h = int(torch.div(center_h / upsampled_size[0], 1. / num_grid))

                # left, top, right, down
                # top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))
                # down_box = min(num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid)))
                # left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))
                # right_box = min(num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid)))
                top_box = max(0, int(torch.div((center_h - half_h) / upsampled_size[0], 1. / num_grid)))
                down_box = min(num_grid - 1, int(torch.div((center_h + half_h) / upsampled_size[0], 1. / num_grid)))
                left_box = max(0, int(torch.div((center_w - half_w) / upsampled_size[1], 1. / num_grid)))
                right_box = min(num_grid - 1, int(torch.div((center_w + half_w) / upsampled_size[1], 1. / num_grid)))

                top = max(top_box, coord_h - 1)
                down = min(down_box, coord_h + 1)
                left = max(coord_w - 1, left_box)
                right = min(right_box, coord_w + 1)

                cate_label[top:(down + 1), left:(right + 1)] = gt_label
                for i in range(top, down + 1):
                    for j in range(left, right + 1):
                        label = int(i * num_grid + j)

                        cur_ins_label = torch.zeros([mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8,
                                                    device=device)
                        cur_ins_label[:seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask
                        ins_label.append(cur_ins_label)
                        ins_ind_label[label] = True
                        grid_order.append(label)
            if len(ins_label) == 0:
                ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
            else:
                ins_label = torch.stack(ins_label, 0)
            ins_label_list.append(ins_label)
            cate_label_list.append(cate_label)
            ins_ind_label_list.append(ins_ind_label)
            grid_order_list.append(grid_order)
        return ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list

    def loss(self, cate_preds, kernel_preds, mask_feat, offset_mask_pred, targets):
        """loss function"""
        # pass
        kernel_lap1_preds, kernel_lap2_preds, kernel_lap3_preds = kernel_preds
        ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list = targets
        # ins
        ins_labels = [torch.cat([ins_labels_level_img
                                 for ins_labels_level_img in ins_labels_level], 0)
                      for ins_labels_level in zip(*ins_label_list)]

        kernel_lap1_preds = [[kernel_preds_level_img.view(kernel_preds_level_img.shape[0], -1)[:,grid_orders_level_img]
                              for kernel_preds_level_img, grid_orders_level_img in
                              zip(kernel_preds_level, grid_orders_level)]
                             for kernel_preds_level, grid_orders_level in zip(kernel_lap1_preds,zip(*grid_order_list))]

        kernel_lap2_preds = [[kernel_preds_level_img.view(kernel_preds_level_img.shape[0], -1)[:,grid_orders_level_img]
                              for kernel_preds_level_img, grid_orders_level_img in
                              zip(kernel_preds_level, grid_orders_level)]
                             for kernel_preds_level, grid_orders_level in zip(kernel_lap2_preds,zip(*grid_order_list))]

        kernel_lap3_preds = [[kernel_preds_level_img.view(kernel_preds_level_img.shape[0], -1)[:,grid_orders_level_img]
                              for kernel_preds_level_img, grid_orders_level_img in
                              zip(kernel_preds_level, grid_orders_level)]
                             for kernel_preds_level, grid_orders_level in zip(kernel_lap3_preds,zip(*grid_order_list))]

        mask_feat_lap1, mask_feat_lap2, mask_feat_lap3 = mask_feat
        offset_mask_lap1, offset_mask_lap2, offset_mask_lap3 = offset_mask_pred

        # generate masks
        ins_pred_list = []
        for b_i, b_kernel_lap1_pred in enumerate(kernel_lap1_preds):
            b_mask_pred = []
            b_kernel_lap2_pred = kernel_lap2_preds[b_i]
            b_kernel_lap3_pred = kernel_lap3_preds[b_i]
            for idx, kernel_lap1_pred in enumerate(b_kernel_lap1_pred):
                kernel_lap2_pred = b_kernel_lap2_pred[idx]
                kernel_lap3_pred = b_kernel_lap3_pred[idx]
                if kernel_lap1_pred.size()[-1] == 0:
                    continue
                cur_mask_feat_lap1 = mask_feat_lap1[idx, ...]
                cur_mask_feat_lap2 = mask_feat_lap2[idx, ...]
                cur_mask_feat_lap3 = mask_feat_lap3[idx, ...]
                cur_offset_mask_lap1 = offset_mask_lap1[idx, ...]
                cur_offset_mask_lap2 = offset_mask_lap2[idx, ...]
                cur_offset_mask_lap3 = offset_mask_lap3[idx, ...]
                N, I = kernel_lap1_pred.shape
                cur_mask_feat_lap1 = cur_mask_feat_lap1.unsqueeze(0)
                cur_mask_feat_lap2 = cur_mask_feat_lap2.unsqueeze(0)
                cur_mask_feat_lap3 = cur_mask_feat_lap3.unsqueeze(0)
                cur_offset_mask_lap1 = cur_offset_mask_lap1.unsqueeze(0)
                cur_offset_mask_lap2 = cur_offset_mask_lap2.unsqueeze(0)
                cur_offset_mask_lap3 = cur_offset_mask_lap3.unsqueeze(0)
                kernel_lap1_pred = kernel_lap1_pred.permute(1, 0).view(I, -1, 3, 3).contiguous()
                kernel_lap2_pred = kernel_lap2_pred.permute(1, 0).view(I, -1, 3, 3).contiguous()
                kernel_lap3_pred = kernel_lap3_pred.permute(1, 0).view(I, -1, 3, 3).contiguous()
                cur_ins_lap1_pred = DynamicDeformConv(cur_mask_feat_lap1, kernel_lap1_pred, cur_offset_mask_lap1,
                                                      stride=1)
                cur_ins_lap2_pred = DynamicDeformConv(cur_mask_feat_lap2, kernel_lap2_pred, cur_offset_mask_lap2,
                                                      stride=1)
                cur_ins_lap3_pred = DynamicDeformConv(cur_mask_feat_lap3, kernel_lap3_pred, cur_offset_mask_lap3,
                                                      stride=1)

                cur_ins_lap3_up = F.interpolate(cur_ins_lap3_pred, scale_factor=2, mode='bilinear',align_corners=False)
                cur_ins_lap2 = cur_ins_lap2_pred + cur_ins_lap3_up
                cur_ins_lap2_up = F.interpolate(cur_ins_lap2, scale_factor=2, mode='bilinear', align_corners=False)
                cur_ins_pred = cur_ins_lap1_pred + cur_ins_lap2_up

                b_mask_pred.append(cur_ins_pred.squeeze(0))
            if len(b_mask_pred) == 0:
                b_mask_pred = None
            else:
                b_mask_pred = torch.cat(b_mask_pred, 0)
            ins_pred_list.append(b_mask_pred)

        ins_ind_labels = [
            torch.cat([ins_ind_labels_level_img.flatten()
                       for ins_ind_labels_level_img in ins_ind_labels_level])
            for ins_ind_labels_level in zip(*ins_ind_label_list)
        ]
        flatten_ins_ind_labels = torch.cat(ins_ind_labels)

        num_ins = flatten_ins_ind_labels.sum()

        # dice loss
        ####################################################################
        # normal dice loss
        loss_ins = []

        for input, target in zip(ins_pred_list, ins_labels):
            if input is None:
                continue
            input = torch.sigmoid(input)
            # loss_ins.append(iou_loss(input, target))
            loss_ins.append(dice_loss(input, target))
        loss_ins_mean = torch.cat(loss_ins).mean()
        ####################################################################
        loss_ins = loss_ins_mean * self.ins_loss_weight

        # cate
        cate_labels = [
            torch.cat([cate_labels_level_img.flatten()
                       for cate_labels_level_img in cate_labels_level])
            for cate_labels_level in zip(*cate_label_list)
        ]
        flatten_cate_labels = torch.cat(cate_labels)

        cate_preds = [
            cate_pred.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            for cate_pred in cate_preds
        ]
        flatten_cate_preds = torch.cat(cate_preds)

        # prepare one_hot
        pos_inds = torch.nonzero(flatten_cate_labels != self.num_classes, as_tuple=False).squeeze(1)

        flatten_cate_labels_oh = torch.zeros_like(flatten_cate_preds)
        flatten_cate_labels_oh[pos_inds, flatten_cate_labels[pos_inds]] = 1

        loss_cate = self.focal_loss_weight * sigmoid_focal_loss_jit(flatten_cate_preds, flatten_cate_labels_oh,
                                                                    gamma=self.focal_loss_gamma,
                                                                    alpha=self.focal_loss_alpha,
                                                                    reduction="sum") / (num_ins + 1)
        return {'loss_ins': loss_ins,
                'loss_cate': loss_cate}

    @staticmethod
    def split_feats(feats):
        """split_feats"""
        return (feats[0],
                F.interpolate(feats[1], scale_factor=0.5, mode='bilinear', align_corners=False),
                feats[2],
                feats[3],
                feats[4],
                F.interpolate(feats[5], size=feats[4].shape[-2:], mode='bilinear', align_corners=False))

    def inference(self, pred_cates, pred_kernels, pred_masks, offset_mask_pred, cur_sizes, images):
        """inference function"""
        pred_lap1_kernels, pred_lap2_kernels, pred_lap3_kernels = pred_kernels
        pred_masks_lap1, pred_masks_lap2, pred_masks_lap3 = pred_masks
        pred_offset_masks_lap1, pred_offset_masks_lap2, pred_offset_masks_lap3 = offset_mask_pred
        assert len(pred_cates) == len(pred_lap1_kernels)

        results = []
        num_ins_levels = len(pred_cates)
        for img_idx in range(len(images)):
            # image size.
            ori_img = images[img_idx]
            height, width = ori_img["height"], ori_img["width"]
            ori_size = (height, width)

            # prediction.
            pred_cate = [pred_cates[i][img_idx].view(-1, self.num_classes).detach()
                         for i in range(num_ins_levels)]
            pred_lap1_kernel = [pred_lap1_kernels[i][img_idx].permute(1, 2, 0).view(-1, self.num_kernels * 9).detach()
                                for i in range(num_ins_levels)]
            pred_lap2_kernel = [pred_lap2_kernels[i][img_idx].permute(1, 2, 0).view(-1, self.num_kernels * 9).detach()
                                for i in range(num_ins_levels)]
            pred_lap3_kernel = [pred_lap3_kernels[i][img_idx].permute(1, 2, 0).view(-1, self.num_kernels * 9).detach()
                                for i in range(num_ins_levels)]

            pred_mask_lap1 = pred_masks_lap1[img_idx, ...].unsqueeze(0)
            pred_mask_lap2 = pred_masks_lap2[img_idx, ...].unsqueeze(0)
            pred_mask_lap3 = pred_masks_lap3[img_idx, ...].unsqueeze(0)

            pred_offset_mask_lap1 = pred_offset_masks_lap1[img_idx, ...].unsqueeze(0)
            pred_offset_mask_lap2 = pred_offset_masks_lap2[img_idx, ...].unsqueeze(0)
            pred_offset_mask_lap3 = pred_offset_masks_lap3[img_idx, ...].unsqueeze(0)

            pred_cate = torch.cat(pred_cate, dim=0)
            pred_lap1_kernel = torch.cat(pred_lap1_kernel, dim=0)
            pred_lap2_kernel = torch.cat(pred_lap2_kernel, dim=0)
            pred_lap3_kernel = torch.cat(pred_lap3_kernel, dim=0)

            pred_kernel = (pred_lap1_kernel, pred_lap2_kernel, pred_lap3_kernel)
            pred_mask = (pred_mask_lap1, pred_mask_lap2, pred_mask_lap3)
            pred_offset_mask = (pred_offset_mask_lap1, pred_offset_mask_lap2, pred_offset_mask_lap3)

            # inference for single image.
            result = self.inference_single_image(pred_cate, pred_kernel, pred_mask, pred_offset_mask,
                                                 cur_sizes[img_idx], ori_size)
            results.append({"instances": result})
        return results

    def inference_single_image(
            self, cate_preds, kernel_preds, seg_preds, offset_mask_preds, cur_size, ori_size
    ):
        """inference_single_image"""
        # overall info.
        h, w = cur_size
        f_h, f_w = seg_preds[0].size()[-2:]
        ratio = math.ceil(h / f_h)
        upsampled_size_out = (int(f_h * ratio), int(f_w * ratio))

        # split kernels.
        kernel_lap1_preds, kernel_lap2_preds, kernel_lap3_preds = kernel_preds
        seg_lap1_preds, seg_lap2_preds, seg_lap3_preds = seg_preds
        offset_mask_lap1_preds, offset_mask_lap2_preds, offset_mask_lap3_preds = offset_mask_preds

        # process.
        inds = (cate_preds > self.score_threshold)
        cate_scores = cate_preds[inds]
        if len(cate_scores) == 0:
            results = Instances(ori_size)
            results.scores = torch.tensor([])
            results.pred_classes = torch.tensor([])
            results.pred_masks = torch.tensor([])
            results.pred_boxes = Boxes(torch.tensor([]))
            return results

        # cate_labels & kernel_preds
        inds = inds.nonzero(as_tuple=False)
        cate_labels = inds[:, 1]
        kernel_lap1_preds = kernel_lap1_preds[inds[:, 0]]
        kernel_lap2_preds = kernel_lap2_preds[inds[:, 0]]
        kernel_lap3_preds = kernel_lap3_preds[inds[:, 0]]

        # trans vector.
        size_trans = cate_labels.new_tensor(self.num_grids).pow(2).cumsum(0)
        strides = kernel_lap1_preds.new_ones(size_trans[-1])

        n_stage = len(self.num_grids)
        strides[:size_trans[0]] *= self.instance_strides[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_ - 1]:size_trans[ind_]] *= self.instance_strides[ind_]
        strides = strides[inds[:, 0]]

        # mask encoding.
        N, I = kernel_lap1_preds.shape
        kernel_lap1_preds = kernel_lap1_preds.view(N, -1, 3, 3).contiguous()
        kernel_lap2_preds = kernel_lap2_preds.view(N, -1, 3, 3).contiguous()
        kernel_lap3_preds = kernel_lap3_preds.view(N, -1, 3, 3).contiguous()

        seg_lap1_preds = DynamicDeformConv(seg_lap1_preds, kernel_lap1_preds, offset_mask_lap1_preds, stride=1)
        seg_lap2_preds = DynamicDeformConv(seg_lap2_preds, kernel_lap2_preds, offset_mask_lap2_preds, stride=1)
        seg_lap3_preds = DynamicDeformConv(seg_lap3_preds, kernel_lap3_preds, offset_mask_lap3_preds, stride=1)

        seg_lap3_preds_up = F.interpolate(seg_lap3_preds, scale_factor=2, mode='bilinear', align_corners=False)
        seg_lap2_preds = seg_lap2_preds + seg_lap3_preds_up
        seg_lap2_preds_up = F.interpolate(seg_lap2_preds, scale_factor=2, mode='bilinear', align_corners=False)
        seg_preds = seg_lap1_preds + seg_lap2_preds_up
        seg_preds = seg_preds.squeeze(0).sigmoid_()

        # mask.
        seg_masks = seg_preds > self.maskness_threshold
        sum_masks = seg_masks.sum((1, 2)).float()

        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            results = Instances(ori_size)
            results.scores = torch.tensor([])
            results.pred_classes = torch.tensor([])
            results.pred_masks = torch.tensor([])
            results.pred_boxes = Boxes(torch.tensor([]))
            return results

        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # maskness.
        seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores

        # sort and keep top nms_pre
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > self.max_before_nms:
            sort_inds = sort_inds[:self.max_before_nms]
        seg_masks = seg_masks[sort_inds, :, :]
        seg_preds = seg_preds[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        if self.nms_type == "matrix":
            matrix
            nms & filter.matrix
            cate_scores = matrix_nms(cate_labels, seg_masks, sum_masks, cate_scores,
                                     sigma=self.nms_sigma, kernel=self.nms_kernel)
            keep = cate_scores >= self.update_threshold
        elif self.nms_type == "mask":
            # original mask nms.
            keep = mask_nms(cate_labels, seg_masks, sum_masks, cate_scores,
                            nms_thr=self.mask_threshold)
        else:
            raise NotImplementedError

        seg_masks = seg_preds > self.mask_threshold
        sum_masks = seg_masks.sum((1, 2)).float()
        keep_sum = sum_masks > 1
        keep = keep * keep_sum

        if keep.sum() == 0:
            results = Instances(ori_size)
            results.scores = torch.tensor([])
            results.pred_classes = torch.tensor([])
            results.pred_masks = torch.tensor([])
            results.pred_boxes = Boxes(torch.tensor([]))
            return results

        seg_preds = seg_preds[keep, :, :]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # sort and keep top_k
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > self.max_per_img:
            sort_inds = sort_inds[:self.max_per_img]
            seg_preds = seg_preds[sort_inds, :, :]
            cate_scores = cate_scores[sort_inds]
            cate_labels = cate_labels[sort_inds]

        # reshape to original size.
        seg_preds = F.interpolate(seg_preds.unsqueeze(0),
                                  size=upsampled_size_out,
                                  mode='bilinear', align_corners=False)[:, :, :h, :w]

        seg_masks = F.interpolate(seg_preds,
                                  size=ori_size,
                                  mode='bilinear', align_corners=False).squeeze(0)

        seg_masks = seg_masks > self.mask_threshold

        # get bbox from mask
        pred_boxes = torch.zeros(seg_masks.size(0), 4)

        for i in range(seg_masks.size(0)):
            mask = seg_masks[i]
            ys, xs = torch.where(mask)
            xmin, ymin, xmax, ymax = 0, 0, 0, 0
            if xs.size(0) != 0:
                xmin = xs.min()
                ymin = ys.min()
                xmax = xs.max()
                ymax = ys.max()
            pred_boxes[i] = torch.tensor([xmin, ymin, xmax, ymax]).float()
            # pred_boxes[i] = torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()]).float()
        pred_boxes = pred_boxes.to(seg_masks.device)

        results = Instances(ori_size)
        results.pred_classes = cate_labels
        results.scores = cate_scores
        results.pred_masks = seg_masks

        results.pred_boxes = Boxes(pred_boxes)

        return results

    def add_bitmasks(self, instances, im_h, im_w):
        """add_bitmasks"""
        for per_im_gt_inst in instances:
            if not per_im_gt_inst.has("gt_masks"):
                continue
            start = int(self.mask_out_stride // 2)
            if isinstance(per_im_gt_inst.get("gt_masks"), PolygonMasks):
                polygons = per_im_gt_inst.get("gt_masks").polygons
                per_im_bitmasks = []
                per_im_bitmasks_full = []
                for per_polygons in polygons:
                    bitmask = polygons_to_bitmask(per_polygons, im_h, im_w)
                    bitmask = torch.from_numpy(bitmask).to(self.device).float()
                    start = int(self.mask_out_stride // 2)
                    bitmask_full = bitmask.clone()
                    bitmask = bitmask[start::self.mask_out_stride, start::self.mask_out_stride]

                    assert bitmask.size(0) * self.mask_out_stride == im_h
                    assert bitmask.size(1) * self.mask_out_stride == im_w

                    per_im_bitmasks.append(bitmask)
                    per_im_bitmasks_full.append(bitmask_full)

                per_im_gt_inst.gt_bitmasks = torch.stack(per_im_bitmasks, dim=0)
                # per_im_gt_inst.gt_bitmasks_full = torch.stack(per_im_bitmasks_full, dim=0)
            else:  # RLE format bitmask
                bitmasks = per_im_gt_inst.get("gt_masks").tensor
                h, w = bitmasks.size()[1:]
                # pad to new size
                bitmasks_full = F.pad(bitmasks, (0, im_w - w, 0, im_h - h), "constant", 0)
                bitmasks = bitmasks_full[:, start::self.mask_out_stride, start::self.mask_out_stride]
                per_im_gt_inst.gt_bitmasks = bitmasks
                per_im_gt_inst.gt_bitmasks_full = bitmasks_full


class LapMaskInsHead(nn.Module):
    """ class LapMaskInsHead """
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        SOLOv2 Instance Head.
        """
        super().__init__()
        # fmt: off
        self.num_classes = cfg.MODEL.LAPMASK.NUM_CLASSES
        self.num_kernels = cfg.MODEL.LAPMASK.NUM_KERNELS
        self.num_grids = cfg.MODEL.LAPMASK.NUM_GRIDS
        self.instance_in_features = cfg.MODEL.LAPMASK.INSTANCE_IN_FEATURES
        self.instance_strides = cfg.MODEL.LAPMASK.FPN_INSTANCE_STRIDES
        self.instance_in_channels = cfg.MODEL.LAPMASK.INSTANCE_IN_CHANNELS  # = fpn.
        self.instance_channels = cfg.MODEL.LAPMASK.INSTANCE_CHANNELS
        # Convolutions to use in the towers
        self.type_dcn = cfg.MODEL.LAPMASK.TYPE_DCN
        self.num_levels = len(self.instance_in_features)
        assert self.num_levels == len(self.instance_strides), \
            print("Strides should match the features.")
        # fmt: on

        head_configs = {"cate": (cfg.MODEL.LAPMASK.NUM_INSTANCE_CONVS,
                                 cfg.MODEL.LAPMASK.USE_DCN_IN_INSTANCE,
                                 False),
                        "kernel_lap1": (cfg.MODEL.LAPMASK.NUM_INSTANCE_CONVS,
                                        cfg.MODEL.LAPMASK.USE_DCN_IN_INSTANCE,
                                        cfg.MODEL.LAPMASK.USE_COORD_CONV),
                        "kernel_lap2": (cfg.MODEL.LAPMASK.NUM_INSTANCE_CONVS,
                                        cfg.MODEL.LAPMASK.USE_DCN_IN_INSTANCE,
                                        cfg.MODEL.LAPMASK.USE_COORD_CONV),
                        "kernel_lap3": (cfg.MODEL.LAPMASK.NUM_INSTANCE_CONVS,
                                        cfg.MODEL.LAPMASK.USE_DCN_IN_INSTANCE,
                                        cfg.MODEL.LAPMASK.USE_COORD_CONV)
                        }

        norm = None if cfg.MODEL.LAPMASK.NORM == "none" else cfg.MODEL.LAPMASK.NORM
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, \
            print("Each level must have the same channel!")
        in_channels = in_channels[0]
        assert in_channels == cfg.MODEL.LAPMASK.INSTANCE_IN_CHANNELS, \
            print("In channels should equal to tower in channels!")

        for head in head_configs:
            tower = []
            num_convs, use_deformable, use_coord = head_configs[head]
            if use_coord:
                chn = self.instance_in_channels + 2
            else:
                chn = self.instance_in_channels
            tower.append(nn.Conv2d(
                chn, self.instance_channels,
                kernel_size=3, stride=1,
                padding=1, bias=norm is None
            ))
            if norm == "GN":
                tower.append(nn.GroupNorm(32, self.instance_channels))
            tower.append(nn.SiLU(inplace=True))
            if head == 'cate':
                tower.append(ConvBlock(self.instance_channels, self.instance_channels, n=num_convs))
            else:
                tower.append(ConvBlock(self.instance_channels, self.instance_channels, n=num_convs - 1))
            if use_deformable:
                tower.append(ModulatedDeformConvWithOff(self.instance_channels, self.instance_channels))
                tower.append(nn.GroupNorm(32, self.instance_channels))
                tower.append(nn.SiLU())
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))

        self.cate_pred = nn.Conv2d(
            self.instance_channels, self.num_classes,
            kernel_size=3, stride=1, padding=1
        )
        self.kernel_lap1_pred = nn.Conv2d(
            self.instance_channels, self.num_kernels * 9,
            kernel_size=3, stride=1, padding=1
        )
        self.kernel_lap2_pred = nn.Conv2d(
            self.instance_channels, self.num_kernels * 9,
            kernel_size=3, stride=1, padding=1
        )
        self.kernel_lap3_pred = nn.Conv2d(
            self.instance_channels, self.num_kernels * 9,
            kernel_size=3, stride=1, padding=1
        )

        for modules in [
            self.cate_tower, self.kernel_lap1_tower, self.kernel_lap2_tower, self.kernel_lap3_tower,
            self.cate_pred, self.kernel_lap1_pred, self.kernel_lap2_pred, self.kernel_lap3_pred
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.LAPMASK.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cate_pred.bias, bias_value)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            pass
        """
        cate_pred = []
        kernel_lap1_pred = []
        kernel_lap2_pred = []
        kernel_lap3_pred = []

        for idx, feature in enumerate(features):
            ins_kernel_feat = feature
            # concat coord
            x_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-1], device=ins_kernel_feat.device)
            y_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-2], device=ins_kernel_feat.device)
            y, x = torch.meshgrid(y_range, x_range)
            y = y.expand([ins_kernel_feat.shape[0], 1, -1, -1])
            x = x.expand([ins_kernel_feat.shape[0], 1, -1, -1])
            coord_feat = torch.cat([x, y], 1)
            ins_kernel_feat = torch.cat([ins_kernel_feat, coord_feat], 1)

            # individual feature.
            kernel_feat = ins_kernel_feat
            seg_num_grid = self.num_grids[idx]
            kernel_feat = F.interpolate(kernel_feat, size=seg_num_grid, mode='bilinear', align_corners=False)
            cate_feat = kernel_feat[:, :-2, :, :]

            # kernel
            kernel_lap1_feat = self.kernel_lap1_tower(kernel_feat)
            kernel_lap1_pred.append(self.kernel_lap1_pred(kernel_lap1_feat))

            kernel_lap2_feat = self.kernel_lap2_tower(kernel_feat)
            kernel_lap2_pred.append(self.kernel_lap2_pred(kernel_lap2_feat))

            kernel_lap3_feat = self.kernel_lap3_tower(kernel_feat)
            kernel_lap3_pred.append(self.kernel_lap3_pred(kernel_lap3_feat))

            # cate
            cate_feat = self.cate_tower(cate_feat)
            cate_pred.append(self.cate_pred(cate_feat))
        return cate_pred, (kernel_lap1_pred, kernel_lap2_pred, kernel_lap3_pred)


class LapMaskHead(nn.Module):
    """LapMaskHead"""

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        SOLOv2 Mask Head.
        """
        super().__init__()
        # fmt: off
        self.mask_on = cfg.MODEL.MASK_ON
        self.num_masks = cfg.MODEL.LAPMASK.NUM_MASKS
        self.mask_in_features = cfg.MODEL.LAPMASK.MASK_IN_FEATURES
        self.mask_in_channels = cfg.MODEL.LAPMASK.MASK_IN_CHANNELS
        self.mask_channels = cfg.MODEL.LAPMASK.MASK_CHANNELS
        self.num_levels = len(input_shape)
        assert self.num_levels == len(self.mask_in_features), \
            print("Input shape should match the features.")
        # fmt: on
        norm = None if cfg.MODEL.LAPMASK.NORM == "none" else cfg.MODEL.LAPMASK.NORM

        self.convs_all_levels = nn.ModuleList()
        for i in range(self.num_levels):
            convs_per_level = nn.Sequential()
            if i == 0:
                conv_tower = list()
                conv_tower.append(nn.Conv2d(
                    self.mask_in_channels, self.mask_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=norm is None
                ))
                if norm == "GN":
                    conv_tower.append(nn.GroupNorm(32, self.mask_channels))
                conv_tower.append(nn.SiLU())
                convs_per_level.add_module('conv' + str(i), nn.Sequential(*conv_tower))
                self.convs_all_levels.append(convs_per_level)
                continue

            for j in range(i):
                if j == 0:
                    chn = self.mask_in_channels + 2 if i == 3 else self.mask_in_channels
                    conv_tower = list()
                    conv_tower.append(nn.Conv2d(
                        chn, self.mask_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=norm is None
                    ))
                    if norm == "GN":
                        conv_tower.append(nn.GroupNorm(32, self.mask_channels))
                    conv_tower.append(nn.SiLU())
                    convs_per_level.add_module('conv' + str(j), nn.Sequential(*conv_tower))
                    upsample_tower = nn.Upsample(
                        scale_factor=2, mode='bilinear', align_corners=False)
                    convs_per_level.add_module(
                        'upsample' + str(j), upsample_tower)
                    continue
                conv_tower = list()
                conv_tower.append(nn.Conv2d(
                    self.mask_channels, self.mask_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=norm is None
                ))
                if norm == "GN":
                    conv_tower.append(nn.GroupNorm(32, self.mask_channels))
                conv_tower.append(nn.SiLU())
                convs_per_level.add_module('conv' + str(j), nn.Sequential(*conv_tower))
                upsample_tower = nn.Upsample(
                    scale_factor=2, mode='bilinear', align_corners=False)
                convs_per_level.add_module('upsample' + str(j), upsample_tower)

            self.convs_all_levels.append(convs_per_level)

        self.conv_pred = nn.Sequential(
            nn.Conv2d(
                self.mask_channels, self.num_masks * 4,
                kernel_size=1, stride=1,
                padding=0, bias=norm is None),
            nn.GroupNorm(32, self.num_masks * 4),
            nn.SiLU()
        )

        deformable_groups = 1
        self.offset_mask_conv_lap1 = nn.Conv2d(
            self.num_masks,
            deformable_groups * 3 * 9,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.offset_mask_conv_lap2 = nn.Conv2d(
            self.num_masks,
            deformable_groups * 3 * 9,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.offset_mask_conv_lap3 = nn.Conv2d(
            self.num_masks,
            deformable_groups * 3 * 9,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.btneck_lap2 = ConvBlock(self.num_masks * 4, self.num_masks * 4, n=1)
        self.btneck_lap3 = ConvBlock(self.num_masks * 4, self.num_masks * 4, n=3)

        self.mask_feat_pred_lap1 = nn.Sequential(
            nn.Conv2d(
                self.num_masks * 4, self.num_masks,
                kernel_size=3, stride=1,
                padding=1, bias=norm is None),
            nn.GroupNorm(16, self.num_masks),
            nn.SiLU()
        )
        self.mask_feat_pred_lap2 = nn.Sequential(
            nn.Conv2d(
                self.num_masks * 4, self.num_masks,
                kernel_size=3, stride=1,
                padding=1, bias=norm is None),
            nn.GroupNorm(16, self.num_masks),
            nn.SiLU()
        )

        self.mask_feat_pred_lap3 = nn.Sequential(
            nn.Conv2d(
                self.num_masks * 4, self.num_masks,
                kernel_size=3, stride=1,
                padding=1, bias=norm is None),
            nn.GroupNorm(16, self.num_masks),
            nn.SiLU()
        )

        for modules in [self.convs_all_levels, self.conv_pred, self.mask_feat_pred_lap1, self.mask_feat_pred_lap2,
                        self.mask_feat_pred_lap3]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, 0)

        nn.init.constant_(self.offset_mask_conv_lap1.weight, 0)
        nn.init.constant_(self.offset_mask_conv_lap1.bias, 0)
        nn.init.constant_(self.offset_mask_conv_lap2.weight, 0)
        nn.init.constant_(self.offset_mask_conv_lap2.bias, 0)
        nn.init.constant_(self.offset_mask_conv_lap3.weight, 0)
        nn.init.constant_(self.offset_mask_conv_lap3.bias, 0)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            pass
        """
        assert len(features) == self.num_levels, \
            print("The number of input features should be equal to the supposed level.")

        # bottom features first.
        feature_add_all_level = self.convs_all_levels[0](features[0])
        for i in range(1, self.num_levels):
            mask_feat = features[i]
            if i == 3:  # add for coord.
                x_range = torch.linspace(-1, 1, mask_feat.shape[-1], device=mask_feat.device)
                y_range = torch.linspace(-1, 1, mask_feat.shape[-2], device=mask_feat.device)
                y, x = torch.meshgrid(y_range, x_range)
                y = y.expand([mask_feat.shape[0], 1, -1, -1])
                x = x.expand([mask_feat.shape[0], 1, -1, -1])
                coord_feat = torch.cat([x, y], 1)
                mask_feat = torch.cat([mask_feat, coord_feat], 1)
            # add for top features.
            feature_add_all_level += self.convs_all_levels[i](mask_feat)

        mask_feat = self.conv_pred(feature_add_all_level)

        mask_feat_down2 = F.interpolate(mask_feat, scale_factor=0.5, mode='bilinear', align_corners=False)
        mask_feat_down4 = F.interpolate(mask_feat_down2, scale_factor=0.5, mode='bilinear', align_corners=False)
        mask_feat_up2 = F.interpolate(mask_feat_down4, scale_factor=2, mode='bilinear', align_corners=False)
        mask_feat_up = F.interpolate(mask_feat_down2, scale_factor=2, mode='bilinear', align_corners=False)

        mask_feat_lap1 = mask_feat - mask_feat_up
        mask_feat_lap2 = mask_feat_down2 - mask_feat_up2

        mask_feat_lap2 = self.btneck_lap2(mask_feat_lap2)
        mask_feat_lap3 = self.btneck_lap3(mask_feat_down4)

        mask_feat_lap1 = self.mask_feat_pred_lap1(mask_feat_lap1)
        mask_feat_lap2 = self.mask_feat_pred_lap2(mask_feat_lap2)
        mask_feat_lap3 = self.mask_feat_pred_lap3(mask_feat_lap3)

        offset_mask_pred_lap1 = self.offset_mask_conv_lap1(mask_feat_lap1)
        offset_mask_pred_lap2 = self.offset_mask_conv_lap2(mask_feat_lap2)
        offset_mask_pred_lap3 = self.offset_mask_conv_lap3(mask_feat_lap3)

        return (mask_feat_lap1, mask_feat_lap2, mask_feat_lap3), (
        offset_mask_pred_lap1, offset_mask_pred_lap2, offset_mask_pred_lap3)
