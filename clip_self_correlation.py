import os
from einops import rearrange
import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from scipy.ndimage import median_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from torchvision.transforms import InterpolationMode
NEAREST = InterpolationMode.NEAREST

# from detectron2.projects.point_rend.point_features import point_sample

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import clip_utils
from configs.dataset_cfg import dataset_info, prompt_templates
from configs.metric import scores

device = "cuda"

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform1(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
    ])

def _transform2():
    return Compose([
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def gt_transform(n_px):
    return Compose([
        Resize(n_px, interpolation=NEAREST),
        CenterCrop(n_px),
        # ToTensor(),
    ])

def get_image_name_list(dataset, image_path):
    if dataset=="CITYS19":
        city_list = os.listdir(image_path)
        image_file_list = []
        for c in city_list:
            new_image_path = image_path+'/'+c
            image_file_list = image_file_list+os.listdir(new_image_path)
    else:
        image_file_list = os.listdir(image_path)
    return image_file_list

def get_image_and_gt(dataset, image_size, image_path, gt_path, file_name):
    # file name
    gt_suffix = '.png'
    if dataset in ["ADEfull","PC459"]:
        gt_suffix = '.tif'
    if dataset == "CITYS19":
        city_name = file_name.split('_')[0]
        image_file = image_path+'/'+city_name+'/'+file_name
        gt_file = gt_path+'/'+city_name+'/'+file_name.replace('leftImg8bit.png','gtFine_labelTrainIds.png')
    else:
        image_file = image_path+'/'+file_name
        gt_file = gt_path+'/'+file_name.split('.')[0]+gt_suffix
    # load image
    img = Image.open(image_file)
    img = _transform1(image_size)(img)
    gt = gt_transform(image_size)(Image.open(gt_file))
    gt = np.array(gt)
    gt = gt.astype(np.int16) # 防止溢出
    gt = torch.tensor(gt).to(device)
    return img, gt

def get_text_features(clip,text_labels):
    text_features = []
    for qw in text_labels:
            query = clip_utils.tokenize([temp(qw) for temp in prompt_templates]).to(device)
            feature = clip.encode_text(query)
            feature /= feature.norm(dim=-1, keepdim=True)
            feature = feature.mean(dim=0)
            feature /= feature.norm()
            text_features.append(feature.unsqueeze(0))
    text_features = torch.cat(text_features, dim=0)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features # c,d

def get_visual_features(clip, images):
    visual_features, attn_weights = clip.encode_image(images)
    # b,hw+1,512     12,b,hw+1,hw+1
    visual_features = visual_features/visual_features.norm(dim=-1, keepdim=True)
    return visual_features, attn_weights

def self_clip(clip, dataset, image_size=224,eps=0.7,min=3):
    # text feature
    cls_text_list = dataset_info[dataset]["labels"]
    text_features = get_text_features(clip, cls_text_list)
    C = text_features.shape[0]
    with_bg = False
    if dataset in ['COCO80_val','VOC21','PC60']: # with background
        with_bg = True
        bg_text_list = dataset_info[dataset]["background"]
        bg_features = get_text_features(clip, bg_text_list)
        BG = bg_features.shape[0]
        text_features = torch.cat([text_features,bg_features],dim=0) # C+bg, 512
    # image
    image_path = dataset_info[dataset]["image_path"]
    gt_path = dataset_info[dataset]["gt_path"]
    image_name_list = get_image_name_list(dataset,image_path)
    # calculate the semantic segmentation
    cal_pred = []
    cal_gt = []
    for pi in tqdm(range(len(image_name_list))):
        ori_images, gts = get_image_and_gt(dataset, image_size, image_path, gt_path, image_name_list[pi])
        ori_h, ori_w = gts.shape[-2:]
        images = _transform2()(ori_images).to(device)
        label_set = torch.unique(gts.reshape(-1))
        # ignore background
        if label_set[-1].float==65535:
            label_set = label_set[:-1]
        elif label_set[-1]==255:
            label_set = label_set[:-1]
        else:
            pass
        if label_set.shape[0]==0: # none gt, ignore
            continue
        # visual features
        patch_window = 16
        h,w = images.shape[-2]//patch_window, images.shape[-1]//patch_window
        visual_features, attn_weights = get_visual_features(clip,images.unsqueeze(0))
        cls_token = visual_features[0,0:1] # 1,512
        patch_tokens = visual_features[0,1:] # hw,512
        cls_weights = attn_weights[-2,0,0:1,1:] # 1,hw
        attn_weights = attn_weights[-2,0,1:,1:] # hw,hw
        # global patch denoise, can replace the dice loss denoise
        attn_weights = attn_weights-cls_weights
        attn_weights[attn_weights<0] = 0 # hw,hw
        # logits and preds
        patch_logits = patch_tokens@text_features.T * clip.logit_scale.exp() # 196, C
        patch_logits = patch_logits.permute(1,0).unsqueeze(0).reshape(1,text_features.shape[0],h,w)
        patch_logits = F.interpolate(
            patch_logits,size=(ori_h, ori_w),mode='bilinear',align_corners=False,
        )[0] # C,H,W
        if dataset=='VOC20':
            cls_logits = cls_token@text_features.T * clip.logit_scale.exp() # 1,c
            patch_logits = patch_logits*cls_logits[0].unsqueeze(1).unsqueeze(1)
        patch_preds = patch_logits.argmax(dim=0) # H,W

        if with_bg:
            patch_preds[patch_preds>=C] = C
            gts[gts==255] = C
        # clusters
        dbscan = DBSCAN(eps=eps, min_samples=min)
        labels = dbscan.fit_predict(attn_weights.detach().cpu().numpy())
        labels = torch.from_numpy(labels).to(device)
        db_label_set = torch.unique(labels)
        if db_label_set.shape[0]==1 and db_label_set[0]==-1:
            # no clusters, continue
            cal_pred.append(patch_preds.cpu().numpy())
            cal_gt.append(gts.cpu().numpy())
            # print('no clusters!')
            continue
        if db_label_set[0]==-1:
            db_label_set = db_label_set[1:]
        # clusters post process
        cluster_gts = []
        for l in range(db_label_set.shape[0]):
            temp_attn = attn_weights[labels==db_label_set[l]] # n,l
            temp_attn = temp_attn.mean(dim=0)
            cluster_gts.append(temp_attn)
        cluster_gts = torch.stack(cluster_gts,dim=0) # n,l
        cluster_gts = cluster_gts.reshape(cluster_gts.shape[0],h,w) # n,h,w
        # smooth
        ratio = 4
        cluster_gts = F.interpolate(
            cluster_gts.unsqueeze(0),size=(ratio*h,ratio*w),mode='bilinear',align_corners=False
        )[0] # n,224,224
        cluster_gts = cluster_gts.detach().cpu().numpy()
        for i in range(cluster_gts.shape[0]):
            cluster_gts[i] = median_filter(cluster_gts[i],size=ratio*2-1)
        cluster_gts = torch.from_numpy(cluster_gts)
        cluster_gts = cluster_gts.to(device)
        cluster_gts = F.interpolate(
            cluster_gts.unsqueeze(0),size=(image_size,image_size),mode='bilinear',align_corners=False
        )[0] # n,H,W
        cluster_gts = cluster_gts.argmax(dim=0)
        # vote
        for gt in range(db_label_set.shape[0]):
            mask_preds = patch_preds[cluster_gts==db_label_set[gt]] # n,
            unique_val, counts = torch.unique(mask_preds, return_counts = True)
            if counts.shape[0]==0:
                continue
            pred_label = unique_val[counts.argmax()]
            # pred_label = cluster_preds[gt]
            patch_preds[cluster_gts==db_label_set[gt]] = pred_label

        if with_bg:
            patch_preds[patch_preds>=C] = C
            gts[gts==255] = C
        cal_pred.append(patch_preds.cpu().numpy())
        cal_gt.append(gts.cpu().numpy())
    if with_bg:
        metric_scores = scores(cal_gt, cal_pred, C+1)
    else:
        metric_scores = scores(cal_gt, cal_pred, C)
    print('dataset:',dataset,'image_size:',image_size, 'eps:',eps,'min:',min)
    print('results:',metric_scores)

def self_clip_test():
    clip_type = "ViT-B/16"
    clip_model, _ = clip_utils.load(clip_type, image_size=224) # origin transforms unused
    clip_model = clip_model.to(device)
    print('load clip success!')
    datasets = ["VOC20","VOC21","COCO80_val","COCO171_val","PC59","PC60","PC459","ADE150","ADEfull"]
    # datasets = ["VOC20","ADE150","ADEfull","COCO171_val","PC59", "PC459"]
    # datasets = ["VOC21", "COCO80_val", "PC60"]
    for d in datasets:
        self_clip(clip_model, d, image_size=224,eps=0.7,min=3)
        self_clip(clip_model, d, image_size=336,eps=1.1,min=7)

if __name__=="__main__":
    with torch.no_grad():
        self_clip_test()
