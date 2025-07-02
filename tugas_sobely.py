import numpy as np

# Mendefinisikan kernel Sobel X dan Y (3x3)
sobel_x_kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

sobel_y_kernel = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
], dtype=np.float32)

k_sobel_size = 3  # Ukuran kernel Sobel
pad_sobel = k_sobel_size // 2  # Padding (1 untuk kernel 3x3)

# Menambahkan padding ke citra (dengan metode reflect)
padded_blurred_for_sobel = np.pad(blurred_final, pad_sobel, mode='reflect')

# Inisialisasi matriks untuk menyimpan gradien X dan Y
sobel_x = np.zeros_like(blurred_final, dtype=np.float32)  # Untuk gradien X
sobel_y = np.zeros_like(blurred_final, dtype=np.float32)  # Untuk gradien Y

# Ukuran citra (rows dan cols)
rows, cols = blurred_final.shape

# Menghitung gradien Sobel X dan Y untuk setiap piksel
for r in range(rows):
    for c in range(cols):
        window_sobel = padded_blurred_for_sobel[r + pad_sobel - 1:r + pad_sobel + 2, c + pad_sobel - 1:c + pad_sobel + 2]

        # Perhitungan sobel_x (detail)
        sum_sobelx = 0.0
        for i in range(k_sobel_size):
            for j in range(k_sobel_size):
                sum_sobelx += window_sobel[i, j] * sobel_x_kernel[i, j]
        sobel_x[r, c] = sum_sobelx

        # Perhitungan sobel_y (detail)
        sum_sobely = 0.0
        for i in range(k_sobel_size):
            for j in range(k_sobel_size):
                sum_sobely += window_sobel[i, j] * sobel_y_kernel[i, j]
        sobel_y[r, c] = sum_sobely
