import cv2

def resample_image(input_path, output_path, resolution='original', width=None, height=None):
    """Mengubah resolusi gambar dengan interpolasi bicubic."""
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Failed to load image: {input_path}")

    h, w = img.shape[:2]
    if resolution == 'original':
        new_w, new_h = w, h
    elif resolution == '720p':
        new_w, new_h = 1280, 720
    elif resolution == '1080p':
        new_w, new_h = 1920, 1080
    elif resolution == 'custom' and width and height:
        new_w, new_h = int(width), int(height)
    else:
        raise ValueError("Invalid resolution parameters")

    # Resample dengan bicubic
    if (new_w, new_h) != (w, h):
        result = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    else:
        result = img

    cv2.imwrite(output_path, result)
    return output_path