import cv2
import numpy as np
from skimage.measure import label, regionprops

def count_pimples(skin_image, mask):
    if skin_image is None:
        return {"pimple_count": 0, "pimple_mask": np.zeros_like(mask)}
    if len(skin_image.shape) == 2:
        skin_image = cv2.cvtColor(skin_image, cv2.COLOR_GRAY2BGR)
    elif skin_image.shape[2] == 1:
        skin_image = cv2.cvtColor(skin_image, cv2.COLOR_GRAY2BGR)
    if skin_image.dtype != np.uint8:
        skin_image = cv2.normalize(skin_image, None, 0, 255, cv2.NORM_MINMAX)
        skin_image = skin_image.astype(np.uint8)
    gray = cv2.cvtColor(skin_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=mask)

    blur_small = cv2.GaussianBlur(gray, (5, 5), 0)
    blur_large = cv2.GaussianBlur(gray, (11, 11), 0)

    dog = cv2.absdiff(blur_small, blur_large)

    _, blob_mask = cv2.threshold(dog, 12, 255, cv2.THRESH_BINARY)
    blob_mask = blob_mask.astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blob_mask = cv2.morphologyEx(blob_mask, cv2.MORPH_OPEN, kernel)
    blob_mask = cv2.morphologyEx(blob_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    centers = []
    pimple_mask = np.zeros_like(mask)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 120 or area > 1200:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.45:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        merged = False
        for mx, my in centers:
            if np.hypot(cx - mx, cy - my) < 25:
                merged = True
                break

        if not merged:
            centers.append((cx, cy))
            cv2.drawContours(pimple_mask, [cnt], -1, 255, -1)

    return {
        "pimple_count": len(centers),
        "pimple_mask": pimple_mask
    }



def detect_acne(skin_image, skin_mask):
    hsv = cv2.cvtColor(skin_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    redness = skin_image[:, :, 2] - (skin_image[:, :, 1] + skin_image[:, :, 0]) / 2
    redness = np.clip(redness, 0, 255).astype(np.uint8)
    redness = cv2.normalize(redness, None, 0, 255, cv2.NORM_MINMAX)
    _, red_mask = cv2.threshold(redness, 150, 255, cv2.THRESH_BINARY)

    gray = cv2.cvtColor(skin_image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = np.abs(lap)
    lap = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, texture_mask = cv2.threshold(lap, 130, 255, cv2.THRESH_BINARY)

    acne_mask = cv2.bitwise_and(red_mask, texture_mask)
    acne_mask = cv2.bitwise_and(acne_mask, skin_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    acne_mask = cv2.morphologyEx(acne_mask, cv2.MORPH_CLOSE, kernel)

    return acne_mask

def filter_acne(acne_mask):
    labeled = label(acne_mask)
    regions = regionprops(labeled)

    final_mask = np.zeros_like(acne_mask)
    acne_count = 0

    for r in regions:
        area = r.area
        perimeter = r.perimeter if r.perimeter > 0 else 1
        circularity = 4 * np.pi * area / (perimeter ** 2)

        if 30 < area < 500 and circularity > 0.4:
            final_mask[labeled == r.label] = 255
            acne_count = acne_count + 1

    return final_mask, acne_count

def acne_severity(acne_count, skin_mask):
    skin_area = np.count_nonzero(skin_mask)
    density = acne_count / max(skin_area, 1)
    
    score = np.clip(density * 5000, 0, 100)
    return round(score, 2)

def analyze_acne(skin_image, skin_mask):
    acne_candidates = detect_acne(skin_image, skin_mask)
    acne_mask, acne_count = filter_acne(acne_candidates)
    severity = acne_severity(acne_count, skin_mask)

    return {
        "acne_count" : acne_count,
        "acne_severity": severity,
        "acne_mask": acne_mask
    }