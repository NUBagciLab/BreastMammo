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

def match_foreground_histogram(src_arr, ref_arr, ratio = 1):
    """Foreground-isolated 16-bit histogram matching using scikit-image,

    supporting arbitrary image sizes and blending control.
    """
    # 1. Automatically generate the foreground masks
    src_mask = src_arr > 0
    ref_mask = ref_arr > 0

    # Fallback if either image foreground is completely empty
    if not np.any(src_mask) or not np.any(ref_mask):
        return src_arr.copy()

    # 2. Extract 1D foreground pixel streams (can be entirely different shapes)
    src_pixels = src_arr[src_mask]
    ref_pixels = ref_arr[ref_mask]

    # 3. Use skimage to match the 1D distribution pools
    # Since these are 1D arrays, we set channel_axis=None
    matched_pixels = match_histograms(src_pixels, ref_pixels, channel_axis=None)

    # 4. Apply the linear blending ratio (alpha)
    blended_pixels = (1.0 - ratio) * src_pixels + ratio * matched_pixels

    # 5. Reconstruct the final image canvas
    output_img = src_arr.copy()
    output_img[src_mask] = np.clip(blended_pixels, 0, 65535).astype(np.uint16)

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