from typing import List

import open_clip
import numpy as np
from sklearn.cluster import DBSCAN
import networkx as nx
import torch
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.memory import retry_if_cuda_oom
from torch import nn
from torch.nn import functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

from .cache_text_encoder import PredefinedOvClassifier
from .self_visual_encoder import FeatureExtractor
from .origin_clip import load
from .utils import get_predefined_templates

# https://github.com/MendelXu/SAN/blob/main/san/model/san.py

@META_ARCH_REGISTRY.register()
class CLIPtrase(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        clip_visual_extractor: nn.Module,
        ov_classifier: PredefinedOvClassifier,
        size_divisibility: int,
        asymetric_input: bool = True,
        clip_resolution: float = 0.5,
        pixel_mean: List[float] = [0.48145466, 0.4578275, 0.40821073],
        pixel_std: List[float] = [0.26862954, 0.26130258, 0.27577711],
        sem_seg_postprocess_before_inference: bool = False,
    ):
        super().__init__()
        self.asymetric_input = asymetric_input
        self.clip_resolution = clip_resolution
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.size_divisibility = size_divisibility
        self.clip_visual_extractor = clip_visual_extractor
        self.ov_classifier = ov_classifier
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
    @classmethod
    def from_config(cls, cfg):
        model, _ = load("ViT-B/16")
        ov_classifier = PredefinedOvClassifier(
            model, templates=get_predefined_templates(cfg.MODEL.CLIPTRASE.CLIP_TEMPLATE_SET)
        )
        clip_visual_extractor = FeatureExtractor(
            model.visual,
            frozen_exclude=[],
        )
        pixel_mean = (0.48145466, 0.4578275, 0.40821073)
        pixel_std = (0.26862954, 0.26130258, 0.27577711)
        pixel_mean = [255.0 * x for x in pixel_mean]
        pixel_std = [255.0 * x for x in pixel_std]

        return {
            "clip_visual_extractor": clip_visual_extractor,
            "ov_classifier": ov_classifier,
            "size_divisibility": cfg.MODEL.CLIPTRASE.SIZE_DIVISIBILITY,
            "asymetric_input": cfg.MODEL.CLIPTRASE.ASYMETRIC_INPUT,
            "clip_resolution": cfg.MODEL.CLIPTRASE.CLIP_RESOLUTION,
            "sem_seg_postprocess_before_inference": cfg.MODEL.CLIPTRASE.SEM_SEG_POSTPROCESS_BEFORE_INFERENCE,
            "pixel_mean": pixel_mean,
            "pixel_std": pixel_std,
        }
    @torch.no_grad()
    def forward(self, batched_inputs):
        # get classifier weight for each dataset
        # !! Could be computed once and saved. It will run only once per dataset.
        with torch.no_grad():
            if "vocabulary" in batched_inputs[0]:
                ov_classifier_weight = (
                    self.ov_classifier.logit_scale.exp()
                    * self.ov_classifier.get_classifier_by_vocabulary(
                        batched_inputs[0]["vocabulary"]
                    )
                )
            else:
                dataset_names = [x["meta"]["dataset_name"] for x in batched_inputs]
                assert (
                    len(list(set(dataset_names))) == 1
                ), "All images in a batch must be from the same dataset."
                ov_classifier_weight = (
                    self.ov_classifier.logit_scale.exp()
                    * self.ov_classifier.get_classifier_by_dataset_name(dataset_names[0])
                )  # C,ndim
            if self.training:
                raise RuntimeError('training-free CLIPtrase has no training mode!')
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)
            clip_input = images.tensor
            if self.asymetric_input:
                clip_input = F.interpolate(
                    clip_input, scale_factor=self.clip_resolution, mode="bilinear"
                )
            ori_h, ori_w = clip_input.shape[-2:]
            h,w = ori_h//16, ori_w//16
            # visual features
            clip_image_features, attn_weights = self.clip_visual_extractor(clip_input)
            cls_weights = attn_weights[-2,0,0:1,1:] # 1,hw
            attn_weights = attn_weights[-2,0,1:,1:] # hw,hw
            # global patch denoise, can replace the dice loss denoise
            attn_weights = attn_weights-cls_weights
            attn_weights[attn_weights<0] = 0 # hw,hw
            # cluster
            dbscan = DBSCAN(eps=1.1, min_samples=7)
            labels = dbscan.fit_predict(attn_weights.detach().cpu().numpy())
            labels = torch.from_numpy(labels).to(attn_weights.device)
            db_label_set = torch.unique(labels)
            cluster_gts = []
            for l in range(db_label_set.shape[0]):
                if db_label_set[l]==-1:
                    continue
                temp_attn = attn_weights[labels==db_label_set[l]] # n,l
                temp_attn = temp_attn.mean(dim=0)
                cluster_gts.append(temp_attn)
            if len(cluster_gts)!=0:
                cluster_gts = torch.stack(cluster_gts,dim=0) # n,l
                cluster_gts = cluster_gts.reshape(cluster_gts.shape[0],h,w) # n,h,w
            else:
                cluster_gts = None
            # logits
            cls_features = clip_image_features[0,0:1,:] # 1,d
            patch_features = clip_image_features[0,1:,:] # hw,d
            text_features = ov_classifier_weight # c,d
            h,w = self.clip_visual_extractor.h, self.clip_visual_extractor.w
            patch_logits = text_features@patch_features.T # c,hw
            patch_logits = patch_logits.reshape(text_features.shape[0],h,w) # c,h,w
            if 'voc' in batched_inputs[0]['meta']['dataset_name']:
                cls_logits = text_features@cls_features.T # c,1
                patch_logits = patch_logits*cls_logits.unsqueeze(1)
            processed_results = []
            for input_per_image, image_size in zip(batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})
                patch_logits = retry_if_cuda_oom(sem_seg_postprocess)(
                    patch_logits, image_size, height, width
                ) # c,h,w
                if cluster_gts!=None:
                    label_num = cluster_gts.shape[0]
                    ratio = 4
                    cluster_gts = F.interpolate(
                        cluster_gts.unsqueeze(0),size=(ratio*h,ratio*w),mode='bilinear',align_corners=False
                    )[0] # n,224,224
                    cluster_gts = cluster_gts.detach().cpu().numpy()
                    for i in range(cluster_gts.shape[0]):
                        cluster_gts[i] = median_filter(cluster_gts[i],size=ratio*2-1)
                    cluster_gts = torch.from_numpy(cluster_gts)
                    cluster_gts = cluster_gts.to(attn_weights.device)
                    cluster_gts = retry_if_cuda_oom(sem_seg_postprocess)(
                        cluster_gts, image_size, height, width
                    ) # n,h,w
                    cluster_gts = cluster_gts.argmax(dim=0) # h,w
                    for gt in range(label_num):
                        mask_preds = patch_logits[:,cluster_gts==gt].mean(dim=-1,keepdim=True)
                        patch_logits[:,cluster_gts==gt] = mask_preds
                    
                processed_results[-1]["sem_seg"] = patch_logits

            return processed_results
        
    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad),
                dtype=gt_masks.dtype,
                device=gt_masks.device,
            )
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    @property
    def device(self):
        return self.pixel_mean.device
