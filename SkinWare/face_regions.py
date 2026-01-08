import cv2
import numpy as np
from SkinWare.skin_scores import compute_skin_scores

def split_face_regions(face_mask):
    h, w = face_mask.shape
    regions = {}

    # Vertical boundaries for face
    forehead_y_end = int(0.30 * h)
    cheek_y_start = int(0.30 * h)
    cheek_y_end = int(0.65 * h)
    chin_y_start = int(0.65 * h)
    
    mid_x = w // 2

    forehead = np.zeros_like(face_mask)
    forehead[: forehead_y_end, :] = face_mask[:forehead_y_end, :]
    regions["forehead"] = forehead

    left_cheek = np.zeros_like(face_mask)
    left_cheek[cheek_y_start : cheek_y_end, : mid_x] = face_mask[cheek_y_start : cheek_y_end, : mid_x]
    regions["left_cheek"] = left_cheek

    right_cheek = np.zeros_like(face_mask)
    right_cheek[cheek_y_start : cheek_y_end, mid_x :] = face_mask[cheek_y_start : cheek_y_end, mid_x :]
    regions["right_cheek"] = right_cheek

    chin = np.zeros_like(face_mask)
    chin[chin_y_start :, :] = face_mask[chin_y_start :, :]
    regions["chin"] = chin

    return regions

def region_scores(skin_image, face_mask):
    regions = split_face_regions(face_mask)
    results = {}
    for name, region_mask in regions.items():
        if cv2.countNonZero(region_mask) < 300:
            continue
        region_skin = cv2.bitwise_and(skin_image, skin_image, mask = region_mask)
        scores = compute_skin_scores(region_skin, region_mask)
        results[name] = scores
    return results
