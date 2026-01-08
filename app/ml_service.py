from SkinWare.pipeline import run_full_pipeline
from SkinWare.inference import run_inference
from SkinWare.skin_scores import compute_skin_scores
from SkinWare.acne_detector import analyze_acne, count_pimples
from SkinWare.face_regions import region_scores
from SkinWare.azure_language import analyze_model_output
from SkinWare.azure_openai_report import generate_explainable_report
import cv2
import os

def analyze_skin_with_ml(image_path: str) -> dict:

    skin, mask, mode = run_full_pipeline(image_path)

    scores = compute_skin_scores(skin, mask)
    pimples = count_pimples(skin, mask)
    acne = analyze_acne(skin, mask)

    regions = None
    if mode == "face":
        regions = region_scores(skin, mask)

    model_result = run_inference(image_path)

    language_meta = analyze_model_output(model_result)

    report = generate_explainable_report(
        model_result,
        language_meta
    )

    return {
        "attributes": scores,
        "pimples": pimples,
        "acne": acne,
        "regions": regions,
        "model_result": model_result,
        "language_meta": language_meta,
        "report": report
    }




# import os
# import shutil
# import tempfile
# import cv2

# from SkinWare.pipeline import run_full_pipeline
# from SkinWare.inference import run_inference
# from SkinWare.skin_scores import compute_skin_scores
# from SkinWare.acne_detector import analyze_acne, count_pimples
# from SkinWare.face_regions import region_scores


# def analyze_skin_with_ml(image_path: str):
#     skin, mask, mode = run_full_pipeline(image_path)

#     scores = compute_skin_scores(skin, mask)
#     pimples = count_pimples(skin, mask)
#     acne = analyze_acne(skin, mask)

#     result = {
#         "scores": scores,
#         "pimple_count": pimples["pimple_count"],
#         "acne": acne,
#         "mode": mode
#     }

#     if mode == "face":
#         result["region_scores"] = region_scores(skin, mask)

#     return result
