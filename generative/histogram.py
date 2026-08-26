# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 16:02:55 2025

@author: pky0507
"""

import os
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import shutil
from seed import seed_everything
import random
from dataset import BreastMammo_density, DenseMammo_density
from skimage.exposure import match_histograms

def match_foreground_histogram(src_arr, ref_arr, ratio=1.0):
    """Foreground-isolated histogram matching supporting arbitrary data types,

    image shapes, and linear blending control.
    """
    src_mask = src_arr > 0
    ref_mask = ref_arr > 0

    if not np.any(src_mask) or not np.any(ref_mask):
        return src_arr.copy()

    src_pixels = src_arr[src_mask]
    ref_pixels = ref_arr[ref_mask]

    matched_pixels = match_histograms(src_pixels, ref_pixels, channel_axis=None)

    # Blend using float operations to prevent premature overflow/underflow
    blended_pixels = (1.0 - ratio) * src_pixels.astype(np.float64) + ratio * matched_pixels.astype(np.float64)

    # Determine dynamic clipping bounds based on src_arr dtype
    src_dtype = src_arr.dtype
    if np.issubdtype(src_dtype, np.integer):
        info = np.iinfo(src_dtype)
        clipped_pixels = np.clip(np.round(blended_pixels), info.min, info.max).astype(src_dtype)
    elif np.issubdtype(src_dtype, np.floating):
        clipped_pixels = blended_pixels.astype(src_dtype)
    else:
        clipped_pixels = blended_pixels.astype(src_dtype)

    output_img = src_arr.copy()
    output_img[src_mask] = clipped_pixels

    return output_img

if __name__ == "__main__":
    seed_everything(42)
    BreastMammo_root = '/dataset/Mammogram/BreastMammo_PNG'
    DenseMammo_root = '/dataset/Mammogram/DenseMammo_PNG'
    alpha_range = [25, 50, 75, 100]
    
    # source_root = BreastMammo_root
    # source = BreastMammo_density(root = BreastMammo_root)
    # ref = DenseMammo_density(root= DenseMammo_root)

    source_root = DenseMammo_root
    ref = BreastMammo_density(root = BreastMammo_root)
    source = DenseMammo_density(root= DenseMammo_root)
    
    generative = os.path.split(source_root)[-1].replace('_PNG', '_Histogram')
    for label in ['Benign', 'Malign']:
        for alpha in alpha_range:
            os.makedirs(os.path.join(generative+str(alpha), label), exist_ok=True)
            shutil.copy2(os.path.join(source_root, label+'_Cases.xlsx'), generative+str(alpha))
    for imgs_path in tqdm(source['image']):
        ref_img = np.asarray(Image.open(random.choice(ref['image'])[0]).resize((512, 512), Image.BICUBIC))
        for source_img_path in imgs_path:
            source_img = np.asarray(Image.open(source_img_path).resize((512, 512), Image.BICUBIC))
            remaining_path, file_name = os.path.split(source_img_path.replace("\\", "/"))
            _, label = os.path.split(remaining_path)
            for alpha in alpha_range:
                synthetic_img = match_foreground_histogram(source_img, ref_img, ratio = alpha/100)
                synthetic_img[source_img<=0] = 0
                cv2.imwrite(os.path.join(generative+str(alpha), label, file_name), synthetic_img)  