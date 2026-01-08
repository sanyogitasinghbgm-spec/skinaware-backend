from PIL import Image

def analyze_skin(file):
    Image.open(file)
    return {
        "dryness": "moderate",
        "redness": "low",
        "uneven_tone": "high",
        "routine": "gentle cleanser + moisturizer"
    }
