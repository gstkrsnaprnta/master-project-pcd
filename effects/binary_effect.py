import cv2

def binary_threshold(input_path, output_path, threshold=127):
    """Mengubah gambar menjadi biner hitam-putih."""
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Failed to load image: {input_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    cv2.imwrite(output_path, binary)
    return output_path