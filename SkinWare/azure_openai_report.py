# Report
import os
from openai import AzureOpenAI

OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-35-turbo")

# client = None
# if OPENAI_ENDPOINT and OPENAI_KEY:
#     client = AzureOpenAI(
#         api_key=OPENAI_KEY,
#         api_version="2024-02-15-preview",
#         azure_endpoint=OPENAI_ENDPOINT
#     )
    
if not OPENAI_ENDPOINT or not OPENAI_KEY:
    raise RuntimeError("Azure OpenAI not configured")

client = AzureOpenAI(
    api_key=OPENAI_KEY,
    api_version="2024-02-15-preview",
    azure_endpoint=OPENAI_ENDPOINT
)

def generate_explainable_report(model_result: dict, language_meta: dict) -> dict:
    """
    Generates a safe, explainable report using Azure OpenAI.
    """

    prompt = f"""
You are a skin analysis assistant.

Model-derived analysis:
{model_result}

Azure Language metadata:
{language_meta}

Rules:
- Educational only
- No medical diagnosis
- Include uncertainty
- Be clear and user-friendly

Return JSON with:
summary, key_observations, confidence, limitations
"""

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return {
    "summary": response.choices[0].message.content
}
