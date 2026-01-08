from pipeline import run_full_pipeline
from inference import run_inference
from skin_scores import compute_skin_scores, skin_detail_scale, skin_view_mode
from acne_detector import analyze_acne, count_pimples
from face_regions import region_scores
import cv2
import tempfile
import os

def analyze_skin_image(image_path: str):
    skin, mask, mode = run_full_pipeline(image_path)

    # scores
    scores = compute_skin_scores(skin, mask)

    pimples = count_pimples(skin, mask)
    acne = analyze_acne(skin, mask)

    model_result = run_inference(image_path)

    response = {
        "scores": scores,
        "pimple_count": pimples["pimple_count"],
        "acne_count": acne["acne_count"],
        "acne_severity": acne["acne_severity"],
        "model_result": model_result,
        "view_mode": skin_view_mode(skin_detail_scale(skin, mask))
    }

    return response
