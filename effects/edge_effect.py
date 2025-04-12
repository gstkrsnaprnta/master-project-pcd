import cv2

def detect_edges(input_path, output_path):
    """Mendeteksi tepi gambar menggunakan Canny."""
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Failed to load image: {input_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    cv2.imwrite(output_path, edges)
    return output_path