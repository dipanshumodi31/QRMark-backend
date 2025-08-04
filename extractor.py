# extractor.py

import cv2
import numpy as np
import pywt
from pyzbar import pyzbar
from PIL import Image
import os

def decode_qr_from_tile(tile, qr_size=(150, 150), tile_index=0):
    # Normalize and preprocess tile
    tile_norm = cv2.normalize(tile, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    sharpened = cv2.filter2D(tile_norm, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
    blurred = cv2.GaussianBlur(sharpened, (5, 5), 0)

    qr_candidate = cv2.resize(blurred, qr_size, interpolation=cv2.INTER_LINEAR)
    qr_binary = cv2.adaptiveThreshold(qr_candidate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)

    debug_dir = "output_images"
    os.makedirs(debug_dir, exist_ok=True)
    debug_path = f"{debug_dir}/qr_candidate_debug_tile{tile_index}.png"
    print(f"[i] Saved extracted QR candidate tile {tile_index} to {debug_path}")

    qr_denoised = cv2.fastNlMeansDenoising(qr_binary, h=10, templateWindowSize=7, searchWindowSize=21)

    kernel_morph = np.ones((3, 3), np.uint8)
    qr_opened = cv2.morphologyEx(qr_denoised, cv2.MORPH_OPEN, kernel_morph)
    qr_closed = cv2.morphologyEx(qr_opened, cv2.MORPH_CLOSE, kernel_morph)

    qr_cleaned = cv2.medianBlur(qr_closed, 3)
    _, qr_final = cv2.threshold(qr_cleaned, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    debug_cleaned_path = f"{debug_dir}/qr_cleaned_debug_tile{tile_index}.png"
    print(f"[i] Cleaned QR tile {tile_index} saved to: {debug_cleaned_path}")

    pil_img = Image.fromarray(qr_final)
    decoded_objects = pyzbar.decode(pil_img)

    if decoded_objects:
        decoded_data = decoded_objects[0].data.decode('utf-8')
        print(f"[+] Decoded QR data from tile {tile_index}: {decoded_data}")
        return decoded_data
    else:
        print(f"[-] No QR code found in tile {tile_index}.")
        return None

def extract_qr_from_cr_hl_subband(watermarked_image_path, qr_size=(150, 150)):
    # Load image
    if isinstance(watermarked_image_path, np.ndarray):
        if watermarked_image_path.shape[2] == 3:
            img_rgb = watermarked_image_path
        else:
            img_rgb = cv2.cvtColor(watermarked_image_path, cv2.COLOR_BGR2RGB)
    else:
        img_bgr = cv2.imread(watermarked_image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Convert to YCrCb and extract Cr channel
    img_ycbcr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    y, cb, cr = cv2.split(img_ycbcr)

    # DWT decomposition of Cr channel
    coeffs = pywt.dwt2(cr, 'haar')
    cA, (cH, cV, cD) = coeffs

    # Split cH into 2x2 tiles
    h, w = cH.shape
    tile_h, tile_w = h // 2, w // 2

    for i in range(2):
        for j in range(2):
            tile = cH[i * tile_h:(i + 1) * tile_h, j * tile_w:(j + 1) * tile_w]
            decoded = decode_qr_from_tile(tile, qr_size, tile_index=(i*2 + j))
            if decoded:
                return decoded

    print("[-] Could not decode any QR tile.")
    return None
