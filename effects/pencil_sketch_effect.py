import cv2
import numpy as np

def pencil_sketch_effect(input_path, output_path):
    """Menciptakan efek sketsa pensil berwarna."""
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Failed to load image: {input_path}")

    # Deteksi tepi
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)
    edges = cv2.divide(gray, gray_blur, scale=255)
    edges = cv2.threshold(edges, 150, 255, cv2.THRESH_BINARY_INV)[1]

    # Lembutkan warna asli
    color_smooth = cv2.GaussianBlur(img, (15, 15), 0)

    # Gabungkan tepi dan warna
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    result = cv2.multiply(color_smooth, edges_bgr, scale=1/255.0)
    cv2.imwrite(output_path, result)
    return output_path