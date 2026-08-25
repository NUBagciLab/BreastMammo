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


def fourier_magnitude_swap(src_arr, ref_arr, L=0.1, ratio=1.0):
    """Advanced 16-bit Fourier Magnitude Swap with low-frequency localization

    and blending ratio control.

    Args:
        src_arr (numpy.ndarray): 16-bit grayscale source image array (uint16).
        ref_arr (numpy.ndarray): 16-bit grayscale reference image array (uint16).
        L (float): Fraction of the center frequencies to swap (0.0 to 1.0).
                   e.g., 0.1 means swapping the inner 10% x 10% square area.
        ratio (float): Alpha blending factor (0.0 to 1.0).
                       1.0 means full reference style replacement,
                       0.0 means keeping original source style.

    Returns:
        numpy.ndarray: Reconstructed 16-bit image array (uint16).
    """
    # 1. Cast arrays to float32 and normalize to [0.0, 1.0]
    src_float = src_arr.astype(np.float32) / 65535.0
    ref_float = ref_arr.astype(np.float32) / 65535.0

    # Ensure spatial dimensions match exactly for frequency grid mapping
    if src_float.shape != ref_float.shape:
        ref_float = cv2.resize(
            ref_float,
            (src_float.shape[1], src_float.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    H, W = src_float.shape

    # 2. Compute 2D FFT and shift the low frequencies to the center
    fft_src = np.fft.fftshift(np.fft.fft2(src_float))
    fft_ref = np.fft.fftshift(np.fft.fft2(ref_float))

    # 3. Extract Magnitude and Phase
    mag_src = np.abs(fft_src)
    phase_src = np.angle(fft_src)
    mag_ref = np.abs(fft_ref)

    # 4. Define the bounding box for the low-frequency center window (L)
    # Calculate the half-widths of the square bounding box
    h_window = int(H * L / 2)
    w_window = int(W * L / 2)

    # Center coordinates of the shifted frequency matrix
    c_h, c_w = H // 2, W // 2

    # Define slice coordinates
    top, bottom = c_h - h_window, c_h + h_window
    left, right = c_w - w_window, c_w + w_window

    # 5. Perform the target-swapping and linear blending within the center window
    # Create a copy of the original source magnitude to act as our baseline canvas
    mag_target = mag_src.copy()

    # Blend the reference magnitude into the targeted source area using the ratio (alpha)
    mag_target[top:bottom, left:right] = (1.0 - ratio) * mag_src[
        top:bottom, left:right
    ] + ratio * mag_ref[top:bottom, left:right]

    # 6. Reconstruct the complex matrix back into the frequency space
    fft_swapped_shifted = mag_target * np.exp(1j * phase_src)

    # 7. Reverse the shift and compute the Inverse 2D FFT
    fft_swapped = np.fft.ifftshift(fft_swapped_shifted)
    img_swapped_float = np.abs(np.fft.ifft2(fft_swapped))

    # 8. Clean boundaries and cast back to native 16-bit layout
    img_swapped_float = np.clip(img_swapped_float, 0.0, 1.0)
    img_swapped_16bit = (img_swapped_float * 65535.0).astype(np.uint16)

    return img_swapped_16bit

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
    
    generative = os.path.split(source_root)[-1].replace('_PNG', '_DFT')
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
                synthetic_img = fourier_magnitude_swap(source_img, ref_img, ratio = alpha/100)
                synthetic_img[source_img<=0] = 0
                cv2.imwrite(os.path.join(generative+str(alpha), label, file_name), synthetic_img)  