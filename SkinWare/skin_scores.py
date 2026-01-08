import cv2
import numpy as np
from skimage.filters import sobel
from skimage.feature import local_binary_pattern
from skimage.measure import shannon_entropy

import cv2
import numpy as np

def compute_porosity(skin_image, mask=None):
    if skin_image is None:
        return 0.0

    if len(skin_image.shape) == 2:
        skin_image = cv2.cvtColor(skin_image, cv2.COLOR_GRAY2BGR)

    if skin_image.dtype != np.uint8:
        skin_image = cv2.normalize(skin_image, None, 0, 255, cv2.NORM_MINMAX)
        skin_image = skin_image.astype(np.uint8)

    gray = cv2.cvtColor(skin_image, cv2.COLOR_BGR2GRAY)

    if mask is None:
        mask = np.ones_like(gray) * 255
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.bitwise_and(gray, gray, mask=mask)
    kernel_size = 9
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    pore_signal = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, pore_mask = cv2.threshold(pore_signal, 20, 255, cv2.THRESH_BINARY)
    pore_mask = cv2.medianBlur(pore_mask, 3) 

    contours, _ = cv2.findContours(
        pore_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    pore_count = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area < 15 or area > 150:
            continue

        perimeter = cv2.arcLength(i, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.7:
            continue
        x, y, w, h = cv2.boundingRect(i)
        aspect_ratio = float(w) / h
        if aspect_ratio < 0.7 or aspect_ratio > 1.3:
            continue

        pore_count += 1

    porosity = min(100.0, pore_count * 2.0)
    return round(porosity, 2)

def skin_view_mode(sds):
    if sds >= 70:
        return "micro"
    return "macro"

def dryness_score(skin_image, mask):
    gray = cv2.cvtColor(skin_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask = mask)

    lbp = local_binary_pattern(gray, P = 8, R = 1, method = "uniform")
    lbp_var = lbp.var()
    contrast = gray.std()
    raw = (lbp_var / 50) + (30 / max(contrast, 1))
    
    score = np.clip(raw * 35, 0, 100)
    return round(score, 3)

def redness_score(skin_image, mask):
    b, g, r = cv2.split(skin_image.astype(np.float32))
    b = cv2.bitwise_and(b, b, mask = mask)
    redness = r - (g + b) / 2
    redness_mean = np.mean(redness)

    score = np.clip((redness_mean / 20) * 100, 0, 100)
    return round(score, 2)

def skin_texture_entropy(skin_image, mask):
    gray = cv2.cvtColor(skin_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask = mask)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    entropy = np.mean(np.abs(lap))
    return entropy

def skin_detail_scale(skin_image, mask):
    skin_pixels = cv2.countNonZero(mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    area_ratio = skin_pixels / total_pixels
    texture_energy = skin_texture_entropy(skin_image, mask)
    texture_norm = min(texture_energy / 15.0, 1.0)
    area_norm = min(area_ratio / 0.8, 1.0)
    sds = (0.6 * texture_norm + 0.4 * area_norm) * 100
    return round(sds, 1)

def pigmentation_score(skin_image):
    gray = cv2.cvtColor(skin_image, cv2.COLOR_BGR2GRAY)
    entropy = shannon_entropy(gray)
    
    mean_luminance = np.mean(gray)
    luminance_deviation = max(0, 128 - mean_luminance)

    raw = (entropy - 3) * 20 + luminance_deviation * 0.5
    score = np.clip(raw, 0, 100)
    return round(score, 2)

def texture_score(skin_image):
    gray = cv2.cvtColor(skin_image, cv2.COLOR_BGR2GRAY)
    edges = sobel(gray)
    edge_energy = np.mean(edges)

    local_std = cv2.blur((gray - gray.mean()) ** 2, (7, 7)).mean() ** 0.5
    raw = (np.log1p(edge_energy * 1000) *20) + (local_std * 0.8)
    score = np.clip(raw, 0, 100)
    return round(score, 3)

def compute_skin_scores(skin_image, mask):
    scores = {}
    scores["dryness"] = dryness_score(skin_image, mask)
    scores["redness"] = redness_score(skin_image, mask)
    scores["pigmentation"] = pigmentation_score(skin_image)
    scores["texture"] = texture_score(skin_image)
    scores["porosity"] = compute_porosity(skin_image, mask)
    return scores