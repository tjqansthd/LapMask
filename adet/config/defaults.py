""" inherit from detectron2.config.dafults """
from detectron2.config.defaults import _C

from detectron2.config import CfgNode as CN

# ---------------------------------------------------------------------------- #
# Additional Configs
# ---------------------------------------------------------------------------- #
_C.MODEL.MOBILENET = False
_C.MODEL.BACKBONE.ANTI_ALIAS = False
_C.MODEL.RESNETS.DEFORM_INTERVAL = 1
_C.INPUT.HFLIP_TRAIN = True
_C.INPUT.CROP.CROP_INSTANCE = True

_C.SOLVER.OPTIMIZER = "SGD"

# ---------------------------------------------------------------------------- #
# Basis Module Options
# ---------------------------------------------------------------------------- #
_C.MODEL.BASIS_MODULE = CN()
_C.MODEL.BASIS_MODULE.NAME = "ProtoNet"
_C.MODEL.BASIS_MODULE.NUM_BASES = 4
_C.MODEL.BASIS_MODULE.LOSS_ON = False
_C.MODEL.BASIS_MODULE.ANN_SET = "coco"
_C.MODEL.BASIS_MODULE.CONVS_DIM = 128
_C.MODEL.BASIS_MODULE.IN_FEATURES = ["p3", "p4", "p5"]
_C.MODEL.BASIS_MODULE.NORM = "SyncBN"
_C.MODEL.BASIS_MODULE.NUM_CONVS = 3
_C.MODEL.BASIS_MODULE.COMMON_STRIDE = 8
_C.MODEL.BASIS_MODULE.NUM_CLASSES = 80
_C.MODEL.BASIS_MODULE.LOSS_WEIGHT = 0.3

# The options for BoxInst, which can train the instance segmentation model with box annotations only
# Please refer to the paper https://arxiv.org/abs/2012.02310
_C.MODEL.BOXINST = CN()
# Whether to enable BoxInst
_C.MODEL.BOXINST.ENABLED = False
_C.MODEL.BOXINST.BOTTOM_PIXELS_REMOVED = 10

_C.MODEL.BOXINST.PAIRWISE = CN()
_C.MODEL.BOXINST.PAIRWISE.SIZE = 3
_C.MODEL.BOXINST.PAIRWISE.DILATION = 2
_C.MODEL.BOXINST.PAIRWISE.WARMUP_ITERS = 10000
_C.MODEL.BOXINST.PAIRWISE.COLOR_THRESH = 0.3

# ---------------------------------------------------------------------------- #
# SwinTransformer backbone
# ---------------------------------------------------------------------------- #
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.ARCH = "Swin_L_22K_384"
_C.MODEL.SWIN.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]

_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.MLP_RATIO = 4
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.DROP_RATE = 0.0
_C.MODEL.SWIN.PATCH_NORM = True
_C.MODEL.SWIN.OUT_INDS = (0, 1, 2, 3)
_C.MODEL.SWIN.FROZEN_STAGES = -1
_C.MODEL.SWIN.USE_CHECKPOINT = False
_C.MODEL.SWIN.ATTN_DROP_RATE = 0.0
_C.MODEL.SWIN.APE = False

# ---------------------------------------------------------------------------- #
# VoVNet backbone
# ---------------------------------------------------------------------------- #
_C.MODEL.VOVNET = CN()
_C.MODEL.VOVNET.CONV_BODY = "V-39-eSE"
_C.MODEL.VOVNET.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]

# Options: FrozenBN, GN, "SyncBN", "BN"
_C.MODEL.VOVNET.NORM = "FrozenBN"
_C.MODEL.VOVNET.OUT_CHANNELS = 256
_C.MODEL.VOVNET.BACKBONE_OUT_CHANNELS = 256
_C.MODEL.VOVNET.STAGE_WITH_DCN = (False, False, False, False)
_C.MODEL.VOVNET.WITH_MODULATED_DCN = False
_C.MODEL.VOVNET.DEFORMABLE_GROUPS = 1

# ---------------------------------------------------------------------------- #
# BiFPN options
# ---------------------------------------------------------------------------- #

_C.MODEL.BiFPN = CN()
# Names of the input feature maps to be used by BiFPN
# They must have contiguous power of 2 strides
# e.g., ["res2", "res3", "res4", "res5"]
_C.MODEL.BiFPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
_C.MODEL.BiFPN.OUT_CHANNELS = 160
_C.MODEL.BiFPN.NUM_REPEATS = 6

# Options: "" (no norm), "GN"
_C.MODEL.BiFPN.NORM = ""

# ---------------------------------------------------------------------------- #
# LapMask Options
# ---------------------------------------------------------------------------- #
_C.MODEL.LAPMASK = CN()

# Instance hyper-parameters
_C.MODEL.LAPMASK.INSTANCE_IN_FEATURES = ["p2", "p2", "p3", "p4", "p5", "p6"]
_C.MODEL.LAPMASK.FPN_INSTANCE_STRIDES = [4, 8, 8, 16, 32, 32]
_C.MODEL.LAPMASK.FPN_SCALE_RANGES = ((1, 48), (24, 96), (48, 192), (96, 384), (192, 768), (384, 2048))
_C.MODEL.LAPMASK.SIGMA = 0.2
# Channel size for the instance head.
_C.MODEL.LAPMASK.INSTANCE_IN_CHANNELS = 256
_C.MODEL.LAPMASK.INSTANCE_CHANNELS = 512
# Convolutions to use in the instance head.
_C.MODEL.LAPMASK.NUM_INSTANCE_CONVS = 4
_C.MODEL.LAPMASK.USE_DCN_IN_INSTANCE = False
_C.MODEL.LAPMASK.TYPE_DCN = 'DCN'
_C.MODEL.LAPMASK.NUM_GRIDS = [52, 44, 36, 24, 16, 12]

# Number of foreground classes.
_C.MODEL.LAPMASK.NUM_CLASSES = 80
_C.MODEL.LAPMASK.NUM_KERNELS = 64
_C.MODEL.LAPMASK.NORM = "GN"
_C.MODEL.LAPMASK.USE_COORD_CONV = True
_C.MODEL.LAPMASK.PRIOR_PROB = 0.01

# Mask hyper-parameters.
# Channel size for the mask tower.
_C.MODEL.LAPMASK.MASK_IN_FEATURES = ["p2", "p3", "p4", "p5"]
_C.MODEL.LAPMASK.MASK_IN_CHANNELS = 256
_C.MODEL.LAPMASK.MASK_CHANNELS = 128
_C.MODEL.LAPMASK.NUM_MASKS = 64

# Test cfg.
_C.MODEL.LAPMASK.NMS_PRE = 500
_C.MODEL.LAPMASK.SCORE_THR = 0.1
_C.MODEL.LAPMASK.UPDATE_THR = 0
_C.MODEL.LAPMASK.MASKNESS_THR = 0.2
_C.MODEL.LAPMASK.MASK_THR = 0.55
_C.MODEL.LAPMASK.MAX_PER_IMG = 100
# NMS type: matrix OR mask.
_C.MODEL.LAPMASK.NMS_TYPE = "matrix"
# Matrix NMS kernel type: gaussian OR linear.
_C.MODEL.LAPMASK.NMS_KERNEL = "gaussian"
_C.MODEL.LAPMASK.NMS_SIGMA = 2

# Loss cfg.
_C.MODEL.LAPMASK.LOSS = CN()
_C.MODEL.LAPMASK.LOSS.FOCAL_USE_SIGMOID = True
_C.MODEL.LAPMASK.LOSS.FOCAL_ALPHA = 0.25
_C.MODEL.LAPMASK.LOSS.FOCAL_GAMMA = 2.0
_C.MODEL.LAPMASK.LOSS.FOCAL_WEIGHT = 1.0
_C.MODEL.LAPMASK.LOSS.DICE_WEIGHT = 3.0
