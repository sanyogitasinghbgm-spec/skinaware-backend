import os
from SkinWare.preprocessing.preprocess import preprocess_image
from SkinWare.models.clip_model import CLIPDescriber
from SkinWare.models.fusion import fuse_results
from SkinWare.models.cnn_model import SkinCNN


def run_inference(image_path):
    prompts = []

    # 🔹 FIX: absolute path based on this file's location
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROMPT_PATH = os.path.join(BASE_DIR, "prompts", "skin_descriptors.txt")

    with open(PROMPT_PATH, "r", encoding="utf-8") as file:
        for line in file:
            prompts.append(line.strip())

    image_tensor = preprocess_image(image_path)

    cnn = SkinCNN()
    dryness, texture, redness, pigmentation = cnn.infer(image_tensor)

    clip_model = CLIPDescriber()
    clip_tags = clip_model.describe(image_path, prompts)

    result = fuse_results(
        dryness,
        texture,
        redness,
        pigmentation,
        clip_tags
    )
    return result
