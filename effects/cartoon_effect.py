import cv2

def create_cartoon_effect(input_path, output_path):
    """Membuat efek kartun dengan smoothing dan tepi."""
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Failed to load image: {input_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
    color = cv2.bilateralFilter(img, 9, 75, 75)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    cv2.imwrite(output_path, cartoon)
    return output_path