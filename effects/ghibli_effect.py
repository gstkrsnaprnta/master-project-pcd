import cv2
import numpy as np
from sklearn.cluster import KMeans

def validate_image(img):
    """Validasi gambar input."""
    h, w, _ = img.shape
    if h < 300 or w < 300:
        return True, "Gambar terlalu kecil (resolusi minimal 300x300 piksel). Hasil mungkin kurang maksimal."
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise_level = np.var(cv2.Laplacian(gray, cv2.CV_64F))
    if noise_level > 1000:
        return False, "Gambar terlalu berisik. Silakan gunakan gambar dengan kualitas lebih baik."
    return True, ""

def preprocess_image(img):
    """Preprocessing komprehensif untuk menangani kasus ekstrem."""
    # Validasi gambar
    is_valid, message = validate_image(img)
    if not is_valid:
        raise ValueError(message)
    
    # Simpan pesan peringatan untuk dikembalikan
    warning = message if message else None

    # Super-resolution sederhana jika resolusi rendah
    h, w, _ = img.shape
    if h < 600 or w < 600:
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Denoising multi-scale
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Normalisasi warna
    img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)

    # Ubah ke HSV untuk analisis pencahayaan
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Adaptive CLAHE
    mean_intensity = np.mean(v)
    clip_limit = 2.0 if mean_intensity < 100 else 1.0
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    v = clahe.apply(v)

    # Gamma correction
    gamma = 1.0
    if mean_intensity < 80:
        gamma = 1.5
    elif mean_intensity > 200:
        gamma = 0.5
    img = img.astype(np.float32) / 255.0
    img = np.power(img, gamma)
    img = (img * 255).astype(np.uint8)

    # Kembalikan ke ruang warna BGR
    hsv = cv2.merge([h, s, v])
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img, warning

def detect_faces(img):
    """Deteksi wajah menggunakan Haar Cascade."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def create_foreground_mask(img):
    """Buat mask untuk memisahkan foreground dan background menggunakan GrabCut."""
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    h, w = img.shape[:2]
    rect = (50, 50, w-50, h-50)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    mask2 = mask2 * 255
    kernel = np.ones((5, 5), np.uint8)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
    return mask2

def create_face_mask(img, faces):
    """Buat mask khusus untuk area wajah."""
    face_mask = np.zeros(img.shape[:2], np.uint8)
    for (x, y, w, h) in faces:
        x = max(0, x - int(w * 0.2))
        y = max(0, y - int(h * 0.3))
        w = int(w * 1.4)
        h = int(h * 1.6)
        cv2.rectangle(face_mask, (x, y), (x+w, y+h), 255, -1)
    return face_mask

def analyze_image(img):
    """Analisis gambar untuk menentukan parameter adaptif."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten()
    hist = hist / hist.sum()
    contrast = np.std(hist)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    s_var = np.var(s)
    k_colors = 8 if s_var > 1000 else 16

    edge_low = 50 if contrast < 0.02 else 100
    edge_high = 150 if contrast < 0.02 else 200

    return k_colors, edge_low, edge_high

def bilateral_filter(img, diameter=15, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(img, diameter, sigma_color, sigma_space)

def color_quantization(img, k=8):
    """Color quantization dengan K-Means dan weighting pada warna dominan."""
    h, w, c = img.shape
    img_2d = img.reshape(-1, c).astype(np.float32)
    
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    weights = hist.flatten()
    weights = weights / (weights.sum() + 1e-6)
    
    bin_indices = np.zeros((h * w,), dtype=np.int32)
    for i in range(h * w):
        b = int(img_2d[i, 0] // 32)
        g = int(img_2d[i, 1] // 32)
        r = int(img_2d[i, 2] // 32)
        bin_idx = r * 64 + g * 8 + b
        bin_indices[i] = bin_idx
    
    pixel_weights = weights[bin_indices]
    pixel_weights = pixel_weights / (pixel_weights.sum() + 1e-6)
    
    kmeans = KMeans(n_clusters=k, random_state=0).fit(img_2d, sample_weight=pixel_weights)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_.astype(np.uint8)
    quantized_img = centers[labels].reshape(h, w, c)
    return quantized_img

def edge_detection(img, low_threshold=100, high_threshold=200, thickness=2):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges_canny = cv2.Canny(gray, low_threshold, high_threshold)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = np.uint8(np.clip(sobel, 0, 255))
    edges = cv2.bitwise_or(edges_canny, sobel)
    kernel = np.ones((thickness, thickness), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    edges = cv2.ximgproc.thinning(edges)
    edges = cv2.GaussianBlur(edges, (3, 3), 0)
    edges = cv2.threshold(edges, 1, 255, cv2.THRESH_BINARY)[1]
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return edges

def adjust_colors(img, brightness=1.0, contrast=1.0, saturation=1.0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    s = s * saturation
    s = np.clip(s, 0, 255)
    hsv = cv2.merge([h, s, v])
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    img = img.astype(np.float32)
    img = img * contrast + brightness
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

def map_to_ghibli_palette(img):
    ghibli_palette = {
        'sky': (135, 206, 235),
        'grass': (124, 252, 0),
        'skin': (255, 218, 185),
        'water': (70, 130, 180),
    }
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    hist_h = cv2.calcHist([h], [0], None, [180], [0, 180])
    dominant_hue = np.argmax(hist_h)
    if 90 <= dominant_hue <= 150:
        target_color = ghibli_palette['grass']
    elif 160 <= dominant_hue <= 240:
        target_color = ghibli_palette['sky']
    else:
        target_color = ghibli_palette['skin']
    img = img.astype(np.float32)
    for i in range(3):
        img[:, :, i] = img[:, :, i] * 0.7 + target_color[i] * 0.3
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def enhance_details(img, mask=None):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    if mask is not None:
        result = img.copy()
        result[mask == 255] = sharpened[mask == 255]
        return result
    return sharpened

def add_soft_glow(img):
    glow = cv2.GaussianBlur(img, (15, 15), 0)
    img = cv2.addWeighted(img, 0.8, glow, 0.2, 0)
    return img

def postprocess_ghibli_colors(img, foreground_mask):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    s_mean = np.mean(s[foreground_mask == 255])
    if s_mean > 150:
        s[foreground_mask == 255] = s[foreground_mask == 255] * 0.8
    h[foreground_mask == 0] = h[foreground_mask == 0] + 10
    h = np.clip(h, 0, 180)
    s = np.clip(s, 0, 255)
    hsv = cv2.merge([h, s, v])
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    img = map_to_ghibli_palette(img)
    return img

def create_ghibli_effect(input_path, output_path, image_type='face', brightness=0, contrast=1.0, saturation=1.5):
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Failed to load image: {input_path}")

    img, warning = preprocess_image(img)
    faces = detect_faces(img)
    has_faces = len(faces) > 0
    foreground_mask = create_foreground_mask(img)
    face_mask = create_face_mask(img, faces) if has_faces else np.zeros_like(foreground_mask)
    k_colors, edge_low, edge_high = analyze_image(img)

    sigma_color_face = 20
    sigma_space_face = 20
    edge_thickness_face = 1
    k_colors_face = 20

    sigma_color_fg = 40 if has_faces else 50
    sigma_space_fg = 40 if has_faces else 50
    edge_thickness_fg = 2
    k_colors_fg = 16 if has_faces else k_colors

    sigma_color_bg = 100
    sigma_space_bg = 100
    edge_thickness_bg = 3
    k_colors_bg = 8

    smoothed_face = bilateral_filter(img, sigma_color=sigma_color_face, sigma_space=sigma_space_face)
    smoothed_fg = bilateral_filter(img, sigma_color=sigma_color_fg, sigma_space=sigma_space_fg)
    smoothed_bg = bilateral_filter(img, sigma_color=sigma_color_bg, sigma_space=sigma_space_bg)

    smoothed = np.zeros_like(img)
    smoothed[face_mask == 255] = smoothed_face[face_mask == 255]
    smoothed[(foreground_mask == 255) & (face_mask == 0)] = smoothed_fg[(foreground_mask == 255) & (face_mask == 0)]
    smoothed[foreground_mask == 0] = smoothed_bg[foreground_mask == 0]

    quantized_face = color_quantization(smoothed_face, k=k_colors_face)
    quantized_fg = color_quantization(smoothed_fg, k=k_colors_fg)
    quantized_bg = color_quantization(smoothed_bg, k=k_colors_bg)

    quantized = np.zeros_like(img)
    quantized[face_mask == 255] = quantized_face[face_mask == 255]
    quantized[(foreground_mask == 255) & (face_mask == 0)] = quantized_fg[(foreground_mask == 255) & (face_mask == 0)]
    quantized[foreground_mask == 0] = quantized_bg[foreground_mask == 0]

    edges_face = edge_detection(img, low_threshold=edge_low, high_threshold=edge_high, thickness=edge_thickness_face)
    edges_fg = edge_detection(img, low_threshold=edge_low, high_threshold=edge_high, thickness=edge_thickness_fg)
    edges_bg = edge_detection(img, low_threshold=edge_low, high_threshold=edge_high, thickness=edge_thickness_bg)

    edges = np.zeros_like(img)
    edges[face_mask == 255] = edges_face[face_mask == 255]
    edges[(foreground_mask == 255) & (face_mask == 0)] = edges_fg[(foreground_mask == 255) & (face_mask == 0)]
    edges[foreground_mask == 0] = edges_bg[foreground_mask == 0]

    adjusted = adjust_colors(quantized, brightness=brightness, contrast=contrast, saturation=saturation)
    edges_mask = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
    edges_mask = cv2.threshold(edges_mask, 1, 255, cv2.THRESH_BINARY_INV)[1]
    result = adjusted.copy()
    result[edges_mask == 0] = [0, 0, 0]

    result = enhance_details(result, foreground_mask)
    result = postprocess_ghibli_colors(result, foreground_mask)
    result = add_soft_glow(result)

    cv2.imwrite(output_path, result)
    return output_path, warning  # Kembalikan path dan peringatan

if __name__ == "__main__":
    input_path = "input.jpg"
    output_path = "output_ghibli.jpg"
    try:
        result, warning = create_ghibli_effect(input_path, output_path)
        print(f"Image processed and saved to {result}")
        if warning:
            print(f"Warning: {warning}")
    except ValueError as e:
        print(f"Error: {e}")