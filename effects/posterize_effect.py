import cv2
import numpy as np

def posterize_image(input_path, output_path, levels=4):
    """Mengurangi jumlah warna untuk efek poster."""
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Failed to load image: {input_path}")
    img = img.astype(np.float32) / 255.0
    img = (np.floor(img * levels) / levels) * 255
    img = img.astype(np.uint8)
    cv2.imwrite(output_path, img)
    return output_path