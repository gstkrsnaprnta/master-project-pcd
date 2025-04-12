import cv2
import numpy as np

def emboss_effect(input_path, output_path):
    """Menciptakan efek emboss 3D."""
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Failed to load image: {input_path}")

    # Konversi ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Kernel emboss
    kernel = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]], dtype=np.float32)
    embossed = cv2.filter2D(gray, -1, kernel)

    # Normalisasi dan tambahkan offset untuk efek 3D
    embossed = cv2.normalize(embossed, None, 0, 255, cv2.NORM_MINMAX)
    embossed = cv2.add(embossed, 128)  # Offset untuk bayangan
    embossed = cv2.cvtColor(embossed, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(output_path, embossed)
    return output_path