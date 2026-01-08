import clip
import torch
from PIL import Image

class CLIPDescriber:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model, self.preprocess = clip.load("ViT-B/32", device = self.device)

    def describe(self, image_path, prompts):
        image = Image.open(image_path)
        image_input = self.preprocess(image)
        image_input = image_input.unsqueeze(0)
        image_input = image_input.to(self.device)

        text_tokens = clip.tokenize(prompts)
        text_tokens = text_tokens.to(self.device)

        torch.set_grad_enabled(False)

        image_features = self.model.encode_image(image_input)
        text_features = self.model.encode_text(text_tokens)

        torch.set_grad_enabled(True)

        similarity = image_features @ text_features.T
        similarity = similarity.softmax(dim = -1)

        top_descriptions = []

        count = 0
        while(count < 2):
            max_value = -1
            max_index = -1
            for i in range(0, len(prompts)):
                score = float(similarity[0][i])
                if score > max_value:
                    if prompts[i] not in top_descriptions:
                        max_value = score
                        max_index = i
            if max_index != -1:
                top_descriptions.append(prompts[max_index])
                count = count + 1
            else:
                break
        return top_descriptions
