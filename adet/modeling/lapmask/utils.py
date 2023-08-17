""" utils function"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.layers import DeformConv, ModulatedDeformConv
from detectron2.layers.deform_conv import modulated_deform_conv


def DynamicDeformConv(input, kernel, offset_mask, stride=1, padding=1, dillation=1, bias=None, groups=1,
                      deformable_groups=1):
    """DynamicDeformConv function"""
    o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
    offset = torch.cat((o1, o2), dim=1)
    mask = mask.sigmoid()
    pred_ins = modulated_deform_conv(
        input, offset, mask, kernel, bias, stride, padding, dillation, groups, deformable_groups)
    return pred_ins


class ModulatedDeformConvWithOff(nn.Module):
    """ class ModulatedDeformConvWithOff function"""
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1,
                 dilation=1, deformable_groups=1):
        super(ModulatedDeformConvWithOff, self).__init__()
        self.offset_mask_conv = nn.Conv2d(
            in_channels,
            deformable_groups * 3 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        """ initialize function"""
        self.dcnv2 = ModulatedDeformConv(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            deformable_groups=deformable_groups,
        )

    def forward(self, input):
        """forward function"""
        x = self.offset_mask_conv(input)
        o1, o2, mask = torch.chunk(x, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        output = self.dcnv2(input, offset, mask)
        return output


# from facebook detectron2
class _NewEmptyTensorOp(torch.autograd.Function):
    """class NewEmptyTensorOp"""
    @staticmethod

    def forward(ctx, x, new_shape):
        """forward function"""
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        """backward function"""
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


def Max(x):
    """
    A wrapper around torch.max in Spatial Attention Module (SAM) to support empty inputs and more features.
    """
    if x.numel() == 0:
        output_shape = [x.shape[0], 1, x.shape[2], x.shape[3]]
        empty = _NewEmptyTensorOp.apply(x, output_shape)
        return empty
    return torch.max(x, dim=1, keepdim=True)[0]


class SpatialAttention(nn.Module):
    """class SpatialAttention"""
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        weight_init.c2_msra_fill(self.conv)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """forward function"""
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = Max(x)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.conv(scale)
        return x * self.sigmoid(scale)


class DillatedConv(nn.Module):
    """class DillatedConv"""
    def __init__(self, in_channel, out_channel, padding=1, dilation=1, bias=True):
        super(DillatedConv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=padding, dilation=dilation)
        self.gn = nn.GroupNorm(16, out_channel)
        self.act = nn.SiLU()

    def forward(self, x):
        """forward function"""
        x = self.conv(x)
        x = self.gn(x)
        x = self.act(x)
        return x


def autopad(k, p=None):  # kernel, padding
    """autopad function"""
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """class Conv function"""
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, group=32):  # ch_in, ch_out, kernel,
        # stride, padding, groups
        """init function"""
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        # self.bn = nn.BatchNorm2d(c2)
        self.norm = nn.GroupNorm(group, c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        torch.nn.init.normal_(self.conv.weight, std=0.01)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        """forward function"""
        return self.act(self.norm(self.conv(x)))

    # def fuseforward(self, x):
    #    return self.act(self.conv(x))


# ASPP Module
class Dilated_bottleNeck(nn.Module):
    """class Dilated_bottleNeck"""

    def __init__(self, in_feat, out_feat):
        super(Dilated_bottleNeck, self).__init__()
        self.aspp_d3 = nn.Sequential(Conv(in_feat, in_feat // 2, 1, 1),
                                     DillatedConv(in_feat // 2, in_feat // 2, padding=3, dilation=3, bias=False))
        self.aspp_d6 = nn.Sequential(Conv(in_feat + in_feat // 2, in_feat // 2, 1, 1),
                                     DillatedConv(in_feat // 2, in_feat // 2, padding=6, dilation=6, bias=False))
        self.aspp_d12 = nn.Sequential(Conv(in_feat * 2, in_feat // 2, 1, 1),
                                      DillatedConv(in_feat // 2, in_feat // 2, padding=12, dilation=12, bias=False))
        self.aspp_d18 = nn.Sequential(Conv(in_feat * 2 + in_feat // 2, in_feat // 2, 1, 1),
                                      DillatedConv(in_feat // 2, in_feat // 2, padding=18, dilation=18, bias=False))
        self.reduction2 = Conv(((in_feat // 2) * 4) + (in_feat), out_feat, 3, 1)

    def forward(self, x):
        """forward function"""
        d3 = self.aspp_d3(x)
        cat1 = torch.cat([x, d3], dim=1)
        d6 = self.aspp_d6(cat1)
        cat2 = torch.cat([cat1, d6], dim=1)
        d12 = self.aspp_d12(cat2)
        cat3 = torch.cat([cat2, d12], dim=1)
        d18 = self.aspp_d18(cat3)
        out = self.reduction2(torch.cat([x, d3, d6, d12, d18], dim=1))
        return out  # 256 x H/32 x W/32


class Bottleneck(nn.Module):
    """class  Bottleneck"""
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, group=32):  # ch_in, ch_out, shortcut, groups, expansion
        """init function"""
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """forward function"""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class ConvBlock(nn.Module):
    """class ConvBlock"""
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, group=32):  # ch_in, ch_out, number, shortcut,
        """init function"""
        # groups, expansion
        super(ConvBlock, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, group=group)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n + 1)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        """forward function"""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPP(nn.Module):
    """class SPP function"""
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        """init function"""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """forward function"""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


def _scale_size(size, scale):
    """Rescale a size by a ratio.
    Args:
        size (tuple[int]): (w, h).
        scale (float): Scaling factor.
    Returns:
        tuple[int]: scaled size.
    """
    w, h = size
    return int(w * float(scale) + 0.5), int(h * float(scale) + 0.5)


interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}


def imresize(img,
             size,
             return_scale=False,
             interpolation='bilinear',
             out=None):
    """Resize image to a given size.
    Args:
        img (ndarray): The input image.
        size (tuple[int]): Target size (w, h).
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos".
        out (ndarray): The output destination.
    Returns:
        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or
            `resized_img`.
    """
    h, w = img.shape[:2]
    resized_img = cv2.resize(
        img, size, dst=out, interpolation=interp_codes[interpolation])
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale


def imresize_like(img, dst_img, return_scale=False, interpolation='bilinear'):
    """Resize image to the same size of a given image.
    Args:
        img (ndarray): The input image.
        dst_img (ndarray): The target image.
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Same as :func:`resize`.
    Returns:
        tuple or ndarray: (`resized_img`, `w_scale`, `h_scale`) or
            `resized_img`.
    """
    h, w = dst_img.shape[:2]
    return imresize(img, (w, h), return_scale, interpolation)


def rescale_size(old_size, scale, return_scale=False):
    """Calculate the new size to be rescaled to.
    Args:
        old_size (tuple[int]): The old size (w, h) of image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image size.
    Returns:
        tuple[int]: The new rescaled image size.
    """
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f'Invalid scale {scale}, must be positive.')
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
    else:
        raise TypeError(
            f'Scale must be a number or tuple of int, but got {type(scale)}')

    new_size = _scale_size((w, h), scale_factor)

    if return_scale:
        return new_size, scale_factor
    else:
        return new_size


def imrescale(img, scale, return_scale=False, interpolation='bilinear'):
    """Resize image while keeping the aspect ratio.
    Args:
        img (ndarray): The input image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image.
        interpolation (str): Same as :func:`resize`.
    Returns:
        ndarray: The rescaled image.
    """
    h, w = img.shape[:2]
    new_size, scale_factor = rescale_size((w, h), scale, return_scale=True)
    rescaled_img = imresize(img, new_size, interpolation=interpolation)
    if return_scale:
        return rescaled_img, scale_factor
    else:
        return rescaled_img


def center_of_mass(bitmasks):
    """center of mass function"""
    _, h, w = bitmasks.size()

    ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks.device)
    xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks.device)

    m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
    m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
    m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)
    center_x = m10 / m00
    center_y = m01 / m00
    return center_x, center_y


def point_nms(heat, kernel=2):
    """point_nms function"""
    # kernel must be 2
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=1)
    keep = (hmax[:, :, :-1, :-1] == heat).float()
    return heat * keep


def matrix_nms(cate_labels, seg_masks, sum_masks, cate_scores, sigma=2.0, kernel='gaussian'):
    """matrix_nms function"""
    n_samples = len(cate_labels)
    if n_samples == 0:
        return []

    seg_masks = seg_masks.reshape(n_samples, -1).float()
    # inter.
    inter_matrix = torch.mm(seg_masks, seg_masks.transpose(1, 0))
    # union.
    sum_masks_x = sum_masks.expand(n_samples, n_samples)
    # iou.
    iou_matrix = (inter_matrix / (sum_masks_x + sum_masks_x.transpose(1, 0) - inter_matrix)).triu(diagonal=1)
    # label_specific matrix.
    cate_labels_x = cate_labels.expand(n_samples, n_samples)
    label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0)).float().triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)

    # IoU decay / soft nms
    delay_iou = iou_matrix * label_matrix

    # matrix nms
    if kernel == 'linear':
        delay_matrix = (1 - delay_iou) / (1 - compensate_iou)
        delay_coefficient, _ = delay_matrix.min(0)
    else:
        delay_matrix = torch.exp(-1 * sigma * (delay_iou ** 2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou ** 2))
        delay_coefficient, _ = (delay_matrix / compensate_matrix).min(0)

    # update the score.
    cate_scores_update = cate_scores * delay_coefficient

    return cate_scores_update


def mask_nms(cate_labels, seg_masks, sum_masks, cate_scores, nms_thr=0.5):
    """mask_nms function"""
    n_samples = len(cate_scores)
    if n_samples == 0:
        return []

    keep = seg_masks.new_ones(cate_scores.shape)
    seg_masks = seg_masks.float()

    for i in range(n_samples - 1):
        if not keep[i]:
            continue
        mask_i = seg_masks[i]
        label_i = cate_labels[i]
        for j in range(i + 1, n_samples, 1):
            if not keep[j]:
                continue
            mask_j = seg_masks[j]
            label_j = cate_labels[j]
            if label_i != label_j:
                continue
            # overlaps
            inter = (mask_i * mask_j).sum()
            union = sum_masks[i] + sum_masks[j] - inter
            if union > 0:
                if inter / union > nms_thr:
                    keep[j] = False
            else:
                keep[j] = False
    return keep
