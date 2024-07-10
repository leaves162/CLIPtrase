# -*- coding: utf-8 -*-
from detectron2.config import CfgNode as CN


def add_our_config(cfg):
    # copied from maskformer2
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # optimizer
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    cfg.SOLVER.WEIGHT_DECAY_EMBED_GROUP = [
        "absolute_pos_embed",
        "positional_embedding",
        "pos_embed",
        "query_embed",
    ]
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0
    cfg.SOLVER.CLIP_MULTIPLIER = 1.0
    cfg.SOLVER.TEST_IMS_PER_BATCH = 1

    # cliptrase
    cfg.MODEL.CLIPTRASE = CN()
    cfg.MODEL.CLIPTRASE.CLIP_TEMPLATE_SET = "vild"
    cfg.MODEL.CLIPTRASE.SIZE_DIVISIBILITY = 32
    cfg.MODEL.CLIPTRASE.ASYMETRIC_INPUT = True
    cfg.MODEL.CLIPTRASE.CLIP_RESOLUTION = 0.5
    cfg.MODEL.CLIPTRASE.SEM_SEG_POSTPROCESS_BEFORE_INFERENCE = True

    # wandb
    cfg.WANDB = CN()
    cfg.WANDB.PROJECT = "cliptrase"
    cfg.WANDB.NAME = None
    # use flash attention
    cfg.MODEL.FLASH = False


