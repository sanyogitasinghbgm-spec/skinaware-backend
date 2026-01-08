# azure_language.py
import os
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

LANG_ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT")
LANG_KEY = os.getenv("AZURE_LANGUAGE_KEY")

# client = None
# if LANG_ENDPOINT and LANG_KEY:
#     client = TextAnalyticsClient(
#         endpoint=LANG_ENDPOINT,
#         credential=AzureKeyCredential(LANG_KEY)
#     ) 

if not LANG_ENDPOINT or not LANG_KEY:
    raise RuntimeError("Azure AI Language not configured")

client = TextAnalyticsClient(
    endpoint=LANG_ENDPOINT,
    credential=AzureKeyCredential(LANG_KEY)
)

def analyze_model_output(result: dict) -> dict:
    if client is None:
        raise RuntimeError("Azure client not initialized")
    scores = result.get("scores", {})

    text = (
        f"dryness: {scores.get('dryness', 0)}, "
        f"texture: {scores.get('texture', 0)}, "
        f"redness: {scores.get('redness', 0)}, "
        f"pigmentation: {scores.get('pigmentation', 0)}"
    )

    response = client.analyze_sentiment([text])[0]

    return {
        "sentiment": response.sentiment,
        "confidence_scores": {
            "positive": response.confidence_scores.positive,
            "neutral": response.confidence_scores.neutral,
            "negative": response.confidence_scores.negative
        }
    }
