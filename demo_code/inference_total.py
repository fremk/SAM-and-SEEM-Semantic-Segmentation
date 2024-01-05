import glob
import os
import warnings
import matplotlib.patches as mpatches
import PIL
import cv2
from PIL import Image, ImageEnhance
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import argparse
import whisper
import numpy as np
import pandas as pd
from xdecoder.BaseModel import BaseModel
from xdecoder import build_model
from utils.distributed import init_distributed
from utils.arguments import load_opt_from_config_files
from utils.constants import COCO_PANOPTIC_CLASSES, classes_cmap,cmap
from utils.util import sam_inference, seem_inference, final_mapping, divide_list_into_k_lists, draw_segmentation
import random
import time
from tasks import *

def parse_option():
    parser = argparse.ArgumentParser('SEEM Demo', add_help=False)
    parser.add_argument('--conf_files', default="configs/seem/seem_focall_lang.yaml", metavar="FILE", help='path to config file', )
    parser.add_argument("--batch", default=0, type=int, help="Batch number")
    parser.add_argument("--nb_batches",default=1, type=int, help="How many batches to divide the main batch")
    args = parser.parse_args()

    return args

'''
build args
'''
args = parse_option()
batch = args.batch
nb_batches = args.nb_batches


if __name__ == "__main__":

    sam_checkpoint = "../segment-anything/sam_vit_h_4b8939.pth"

    model_type = "vit_h"

    device = "cuda:0"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    #        points_per_side=64,
    mask_generator = SamAutomaticMaskGenerator(sam,
            points_per_side=32,
            points_per_batch=64,
            pred_iou_thresh = 0.88,
            stability_score_thresh = 0.95,
            stability_score_offset  = 1.0,
            box_nms_thresh = 0.7,
            crop_n_layers = 0,
            crop_nms_thresh= 0.7,
            crop_overlap_ratio= 512 / 1500,
            crop_n_points_downscale_factor = 1,
            min_mask_region_area = 50,
            output_mode = "binary_mask"
        )

    output_path="./output/"

    file1 = open('./input/image_path.txt', 'r')

    images = file1.readlines()
    for i in range(len(images)-1):
        images[i]=images[i][:-1]


    images.sort()
    divided_lists = divide_list_into_k_lists(images, nb_batches)

    images=divided_lists[batch]
    
    #Sam + Seem
    start_time = time.time()
    for image in images:

        image_name=image.rsplit('/',1)[1][:-4]
        print(' ')
        print('Image name: '+image_name)
        initial_image=np.array(Image.open(image))
        # print('Shape',initial_image.shape)
        if(len(initial_image.shape)==2):
            initial_image=np.dstack((initial_image,initial_image,initial_image))
        
        mask = sam_inference(initial_image,mask_generator)


        result_mask, conf_mask, seem_semantic = seem_inference(initial_image, mask)



        plt.imshow(initial_image)
        draw_segmentation(final_mapping(np.array(result_mask)),a=1)
        # plt.show()
        break

        # Uncomment/modify these lines if you want to save the pipeline annotation, seem annotation as well as the logits (scores) 
        
        # result_mask.save(output_path+image_name+'.png')
        # seem_semantic.save(output_path+'seem/'+image_name+'.png')
        # np.savez(output_path+'scores/'+image_name+'.npz',conf_mask.astype(np.float16))
        # # break
    end_time=time.time()
    time=end_time-start_time
    
    hours, remainder = divmod(time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"Time elapsed: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")





