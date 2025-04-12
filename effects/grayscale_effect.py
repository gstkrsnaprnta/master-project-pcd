import cv2

def convert_to_grayscale(input_path, output_path):
    """Mengubah gambar berwarna menjadi grayscale."""
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Failed to load image: {input_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(output_path, gray)
    return output_path