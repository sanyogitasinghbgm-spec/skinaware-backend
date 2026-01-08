from pipeline import run_full_pipeline
from inference import run_inference
from skin_scores import compute_skin_scores, skin_detail_scale, skin_view_mode
from acne_detector import analyze_acne, count_pimples
from face_regions import region_scores
from azure_language import analyze_model_output
from azure_openai_report import generate_explainable_report
import cv2


def main():
    input_image = "sample_skin4.jpg"
    processed_image = "final_skin.jpg"
    # Preprocess
    skin, mask, mode = run_full_pipeline(input_image)
    cv2.imwrite(processed_image, skin)
    print("Preprocessing done, saved as final_skin.jpg")

    # Image scale analysis
    sds = skin_detail_scale(skin, mask)
    view_mode = skin_view_mode(sds)
    print("\nSkin View Mode:", view_mode.upper())
    print("Skin Detail Scale:", sds, "/ 100")

    # Skin scores
    scores = compute_skin_scores(skin, mask)
    print("\nSkin Analysis Scores:")
    for key, value in scores.items():
        if view_mode == "micro" and key in ["texture", "redness", "porosity"]:
            print(f"{key.capitalize()} : {value} / 100 (micro - close-up sensitive)")
        else:
            print(f"{key.capitalize()} : {value} / 100")

    # Acne detect
    pimples = count_pimples(skin, mask)
    print("\nPimple Count:", pimples["pimple_count"])

    acne = analyze_acne(skin, mask)
    print("\nAcne Analysis:")
    print("Acne count:", acne["acne_count"])
    print("Acne severity:", acne["acne_severity"], "/ 100")

    # Facial region scores
    if mode == "face":
        region_s = region_scores(skin, mask)
        print("\nRegion-wise Facial Scores:")
        for region, scores in region_s.items():
            print(f"\n{region.upper()}")
            for key, value in scores.items():
                print(f" {key.capitalize()}: {value} / 100")
    else:
        print("\nRegion-wise Facial Scores: Skipped (Not a face image)")

    # Inference
    print("\nRunning ML inference...")
    model_result = run_inference(processed_image)

    print("\nModel Results:")
    print(model_result)

    # Azure AI Language
    print("\nAnalyzing model output with Azure AI Language...")
    language_meta = analyze_model_output(model_result)

    print("\nAzure Language Analysis:")
    print(language_meta)

    # Azure Foundry
    print("\nGenerating explainable report with Azure OpenAI...")
    final_report = generate_explainable_report(
        model_result,
        language_meta
    )

    print("\nAzure OpenAI Explainable Report:")
    print(final_report["explainable_report"])


if __name__ == "__main__":
    main()