import cv2
import numpy as np

def oil_painting_effect(input_path, output_path, radius=5, levels=10):
    """Menciptakan efek lukisan cat minyak."""
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Failed to load image: {input_path}")

    # Bilateral filter untuk smoothing
    img_smooth = cv2.bilateralFilter(img, d=radius*2, sigmaColor=75, sigmaSpace=75)

    # Kuantisasi warna
    img_float = img_smooth.astype(np.float32) / 255.0
    img_quantized = (np.floor(img_float * levels) / levels) * 255
    img_quantized = img_quantized.astype(np.uint8)

    # Deteksi tepi untuk efek kuas
    gray = cv2.cvtColor(img_quantized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Gabungkan tepi dan warna
    result = cv2.addWeighted(img_quantized, 0.8, edges, 0.2, 0)
    cv2.imwrite(output_path, result)
    return output_path