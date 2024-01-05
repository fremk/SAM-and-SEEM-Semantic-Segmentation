# --------------------------------------------------------
# SEEM -- Segment Everything Everywhere All At Once
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from utils.visualizer import Visualizer
from detectron2.utils.colormap import random_color
from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks
from xdecoder.language.loss import vl_similarity
from utils.constants import COCO_PANOPTIC_CLASSES, classes_COCO_match
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import io
import cv2
import os
import glob
import subprocess
from PIL import Image
import random

t = []
t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
transform = transforms.Compose(t)
metadata = MetadataCatalog.get('coco_2017_train_panoptic')
all_classes = [name.replace('-other','').replace('-merged','') for name in COCO_PANOPTIC_CLASSES] + ["others"]
colors_list = [(np.array(color['color'])/255).tolist() for color in COCO_CATEGORIES] + [[1, 1, 1]]

def interactive_infer_image(model, audio_model, image, tasks, mask=np.zeros((720,1280)), refimg=None, reftxt=None, audio_pth=None, video_pth=None):

    image_ori=image # my modified version
    if (mask==np.zeros((720*1280))):
        pass
    else:
        mask_ori = Image.fromarray(mask).convert('RGB')

    
    width = image_ori.size[0]
    height = image_ori.size[1]
    image_ori = np.asarray(image_ori)
    visual = Visualizer(image_ori, metadata=metadata)
    images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()

    data = {"image": images, "height": height, "width": width}
    if len(tasks) == 0:
        tasks = ["Panoptic"]
    
    model.model.task_switch['spatial'] = False
    model.model.task_switch['visual'] = False
    model.model.task_switch['grounding'] = False
    model.model.task_switch['audio'] = False

    stroke = None
    if 'Stroke' in tasks:
        model.model.task_switch['spatial'] = True
        mask_ori = np.asarray(mask_ori)[:,:,0:1].copy()
        mask_ori = torch.from_numpy(mask_ori).permute(2,0,1)[None,]
        mask_ori = (F.interpolate(mask_ori, (height, width), mode='bilinear') > 0)
        data['stroke'] = mask_ori

    batch_inputs = [data]
    if 'Panoptic' in tasks:
        model.model.metadata = metadata
        results = model.model.evaluate(batch_inputs)
        pano_seg = results[-1]['panoptic_seg'][0]
        pano_seg_info = results[-1]['panoptic_seg'][1]
        result_segmentation_arg=np.array(results[-1]['sem_seg'].permute(1,2,0).cpu())

        new_proba=result_segmentation_arg

        imcop=np.argmax(result_segmentation_arg,axis=-1).astype(np.int32)
        original=[]
        matched=[]

        for i in np.unique(imcop):
            original.append(i)
            keys = [k for k, v in classes_COCO_match.items() if i in v]
            matched.append(int(keys[0]))

        overlap=[]
        for j in range(len(original)):

            if original[j] in matched:

                overlap.append(original[j])
                
                imcop[imcop==original[j]]=original[j]+300
                original[j]=original[j]+300
            else:
                imcop[imcop==original[j]]=matched[j]
        for un in np.unique(imcop):
            if un >= 300:
                for f in range(len(original)):
                    if original[f]==un:
                        imcop[imcop==un]=matched[f]
        im=imcop

        return im, new_proba

    else:
        results,image_size,extra = model.model.evaluate_demo(batch_inputs)

    if 'Stroke' in tasks:
        v_emb = results['pred_maskembs']
        s_emb = results['pred_pspatials']
        pred_masks = results['pred_masks']

        pred_logits = v_emb @ s_emb.transpose(1,2)

        logits_idx_y = pred_logits[:,:,0].max(dim=1)[1]
        logits_idx_x = torch.arange(len(logits_idx_y), device=logits_idx_y.device)

        logits_idx = torch.stack([logits_idx_x, logits_idx_y]).tolist()

        pred_masks_pos = pred_masks[logits_idx]


        pred_class = results['pred_logits'][logits_idx].max(dim=-1)[1]


    pred_masks_pos = (F.interpolate(pred_masks_pos[None,], image_size[-2:], mode='bilinear')[0,:,:data['height'],:data['width']] > 0.0).float().cpu().numpy()

    texts = [all_classes[pred_class[0]]]

    for idx, mask1 in enumerate(pred_masks_pos):
        out_txt = texts[idx] if 'Text' not in tasks else reftxt
        demo = visual.draw_binary_mask(mask1, color=colors_list[pred_class[0]%133], text=out_txt)
    res = demo.get_image()
    torch.cuda.empty_cache()

    if(out_txt=='others'):
        keys=255
    else:
        keys=[k for k, v in classes_COCO_match.items() if pred_class[0]%133 in v][0]

    new_proba=np.zeros((height,width,28))

    return new_proba , int(keys)