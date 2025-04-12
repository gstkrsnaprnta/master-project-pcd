import cv2
import numpy as np

def watercolor_effect(input_path, output_path):
    """Menciptakan efek lukisan cat air."""
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Failed to load image: {input_path}")

    # Smoothing dengan bilateral filter
    img_smooth = cv2.bilateralFilter(img, d=15, sigmaColor=100, sigmaSpace=100)

    # Kuantisasi warna
    img_float = img_smooth.astype(np.float32) / 255.0
    img_quantized = (np.floor(img_float * 8) / 8) * 255
    img_quantized = img_quantized.astype(np.uint8)

    # Tambahkan tekstur noise
    noise = np.random.normal(0, 10, img_quantized.shape).astype(np.uint8)
    result = cv2.add(img_quantized, noise)
    result = np.clip(result, 0, 255)

    # Deteksi tepi halus
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    result = cv2.addWeighted(result, 0.9, edges, 0.1, 0)

    cv2.imwrite(output_path, result)
    return output_path