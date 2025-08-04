# embedder.py

import cv2
import numpy as np
import pywt
from PIL import Image
import qrcode
from io import BytesIO

def generate_qr(data, size=(150, 150)):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=4,
        border=0,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("L")
    img = img.resize(size, Image.NEAREST)
    _, qr_binary = cv2.threshold(np.array(img), 128, 255, cv2.THRESH_BINARY)
    return qr_binary

def embed_qr_in_cr_hl_subband(host_image_path, qr_data, output_image_path, alpha=12):
    if isinstance(host_image_path, Image.Image):
        host_rgb = np.array(host_image_path.convert("RGB"))
    else:
        host_rgb = cv2.imread(host_image_path)

    host_ycbcr = cv2.cvtColor(host_rgb, cv2.COLOR_RGB2YCrCb)
    y, cb, cr = cv2.split(host_ycbcr)

    h, w = cr.shape
    qr_size = (150, 150)
    qr_img = generate_qr(qr_data, qr_size)

    coeffs = pywt.dwt2(cr, 'haar')
    cA, (cH, cV, cD) = coeffs

    qr_resized = cv2.resize(qr_img, (cH.shape[1], cH.shape[0]), interpolation=cv2.INTER_NEAREST)
    qr_normalized = (qr_resized.astype(np.float32) / 255.0 - 0.5) * 2
    qr_tile = np.tile(qr_normalized, (2, 2))  # Repeat twice in both dimensions
    qr_tile = cv2.resize(qr_tile, (cH.shape[1], cH.shape[0]), interpolation=cv2.INTER_NEAREST)
    cH_watermarked = cH + alpha * qr_tile

    coeffs_watermarked = cA, (cH_watermarked, cV, cD)
    cr_watermarked = pywt.idwt2(coeffs_watermarked, 'haar')

    cr_watermarked = cv2.resize(cr_watermarked, (w, h), interpolation=cv2.INTER_LINEAR)
    cr_watermarked = np.clip(cr_watermarked, 0, 255).astype(np.uint8)

    ycbcr_watermarked = cv2.merge((y, cb, cr_watermarked))
    rgb_watermarked = cv2.cvtColor(ycbcr_watermarked, cv2.COLOR_YCrCb2RGB)

    if isinstance(output_image_path, BytesIO):
        is_success, buffer = cv2.imencode(".png", cv2.cvtColor(rgb_watermarked, cv2.COLOR_RGB2BGR))
        output_image_path.write(buffer.tobytes())
        output_image_path.seek(0)
    else:
        cv2.imwrite(output_image_path, cv2.cvtColor(rgb_watermarked, cv2.COLOR_RGB2BGR))

    print(f"[+] QR embedded successfully and saved to {output_image_path}")
