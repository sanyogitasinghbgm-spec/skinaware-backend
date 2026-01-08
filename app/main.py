from fastapi import FastAPI, UploadFile, File, HTTPException
from app.database import users_collection, analysis_collection, feedback_collection, products_collection
from app.auth import hash_password, verify_password, create_token
from app.ml_model import analyze_skin
from bson import ObjectId
from jose import jwt
import os
from app.cloudinary_service import upload_image
from app.crud import get_history
from app.crud import save_analysis
from app.crud import get_history, delete_single_history, delete_all_history
from app.crud import get_recommended_products
from app.ml_service import analyze_skin_with_ml
from app.models import UserCreate
# from skinware.ml_service import analyze_skin_image
import tempfile
import shutil
import numpy as np
from SkinWare.azure_language import analyze_model_output
from SkinWare.azure_openai_report import generate_explainable_report
#from app.utils.serializer import mongo_safe
import json
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="SkinAware API")
# 🔥 CORS FIX (MANDATORY)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite frontend
        "http://localhost:3000",
        "https://skinaware-backend.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#-------Default---------
@app.get("/")
def root():
    return {"message": "SkinAware Backend Running"}


# -------- SIGNUP --------
@app.post("/signup")
def signup(user: UserCreate):
    user_dict = user.model_dump()          # ✅ convert to dict
    user_dict["password"] = hash_password(
        user_dict["password"]
    )
    # user["password"] = hash_password(user["password"])
    users_collection.insert_one(user_dict)

    return {"message": "User created"}

# -------- LOGIN --------
@app.post("/login")
def login(user: dict):
    db_user = users_collection.find_one({"email": user["email"]})
    if not db_user or not verify_password(user["password"], db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_token(db_user["_id"])
    return {"access_token": token}

# -------- ANALYZE --------
def make_mongo_safe(obj):
    if isinstance(obj, dict):
        return {k: make_mongo_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_mongo_safe(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj


def clean_llm_text(text: str):
    if not text:
        return ""
    return (
        text
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

def convert_numpy(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    return obj

from app.cloudinary_service import upload_image


@app.post("/analyze")
def analyze(file: UploadFile = File(...), token: str = ""):
    payload = jwt.decode(token, os.getenv("JWT_SECRET_KEY"), algorithms=["HS256"])
    user_id = payload["sub"]

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        shutil.copyfileobj(file.file, temp)
        temp_path = temp.name
    file.file.close()

    # Upload to Cloudinary
    image_url = upload_image(temp_path)

    # ML Analysis
    ml_result = analyze_skin_with_ml(temp_path)
    ml_result["acne"].pop("acne_mask", None)
    ml_result["pimples"].pop("pimple_mask",None)
    # safe_result = make_mongo_safe(result)
    
    # Azure Language
    language_meta = analyze_model_output(ml_result)
   
     # Azure OpenAI Report
    raw_report = generate_explainable_report(ml_result, language_meta)
    
    # Clean report summary
    clean_summary = clean_llm_text(raw_report.get("summary"))
    try:
        report_data = json.loads(clean_summary)
    except Exception:
        report_data = {"summary": clean_summary}
    response = {
        "analysis_id": None, 
        "image_url": image_url,

        "attributes": convert_numpy(ml_result.get("attributes", {})),

        "acne": {
            "count": int(ml_result.get("acne", {}).get("acne_count", 0)),
            "severity":float(ml_result.get("acne", {}).get("acne_severity", 0))
        },

        "pimples": {
            "count": int(ml_result.get("pimples", {}).get("pimple_count", 0))
        },

        "overall_assessment": {
            "concern_level": ml_result.get("model_result", {}).get("concern_level", "low"),
            "summary": report_data
        }
    }

    # final_result = {
    #     "ml_result": ml_result,
    #     "language_meta": language_meta,
    #     "report": report
    # }

    # Mongo safe
    safe_result = make_mongo_safe(response)

    insert_result = analysis_collection.insert_one({
        "user_id": ObjectId(user_id),
        "result": safe_result,
        "image_url" : image_url,
        "created_at": datetime.utcnow()
    })
    shutil.os.remove(temp_path)
    response["analysis_id"] = str(insert_result.inserted_id)
    return response
    # return { 
    #     "analysis_id": str(insert_result.inserted_id),
    #     "result": safe_result,
    #     "image_url": image_url
    # }

    # #  Save in MongoDB
    # analysis_doc = {
    #     "user_id": ObjectId(user_id),
    #     "result": safe_result,
    #     "image_url": image_url
    # }
    # insert_result = analysis_collection.insert_one(analysis_doc)
    # shutil.os.remove(temp_path)
    # # 5️⃣ Return response with analysis_id
    # return {
    #     "analysis_id": str(insert_result.inserted_id),
    #     "result": safe_result,
    #     "image_url": image_url
    # }

# @app.post("/analyze")
# def analyze(file: UploadFile = File(...), token: str = ""):
#     payload = jwt.decode(token, os.getenv("JWT_SECRET_KEY"), algorithms=["HS256"])
#     user_id = payload["sub"]

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
#         shutil.copyfileobj(file.file, temp)
#         temp_path = temp.name

#     result = analyze_skin_with_ml(temp_path)
#     # analysis_collection.insert_one({
#     #     "user_id": ObjectId(user_id),
#     #     "result": result
#     # })

#     safe_result = make_mongo_safe(result)

#     analysis_collection.insert_one({
#         "user_id": ObjectId(user_id),
#         "result": safe_result
#     })


#     return safe_result


# @app.post("/analyze")
# def analyze(file: UploadFile = File(...), token: str = ""):
#     payload = jwt.decode(token, os.getenv("JWT_SECRET_KEY"), algorithms=["HS256"])
#     user_id = payload["sub"]

#     image_url = upload_image(file.file)
    
#     result = analyze_skin(file.file)
    
#     analysis_id = save_analysis(
#         user_id=user_id,
#         result=result,
#         image_url=image_url
#     )

#     return {
#         "analysis_id": analysis_id,
#         "result": result,
#         "image_url": image_url
#     }

    # analysis_collection.insert_one({
    #     "user_id": ObjectId(user_id),
    #     "result": result
    # })

    #return result

# -------- HISTORY --------
@app.get("/history")
def history(token: str = ""):
    payload = jwt.decode(token, os.getenv("JWT_SECRET_KEY"), algorithms=["HS256"])
    user_id = payload["sub"]

    records = analysis_collection.find(
        {"user_id": ObjectId(user_id)}
    ).sort("created_at", -1)

    history_list = []
    
    for r in records:
        result = r.get("result", {})
        result.pop("analysis_id", None)
        result.pop("image_url", None)
        history_list.append({
            "analysis_id": str(r["_id"]),
            "image_url": r.get("image_url"),
            "created_at": r.get("created_at"),
            "result": r.get("result")
            
        })

    return history_list



# @app.get("/history")
# def history(token: str = ""):
#     payload = jwt.decode(token, os.getenv("JWT_SECRET_KEY"), algorithms=["HS256"])
#     user_id = payload["sub"]
#     return get_history(user_id)
#     #records = analysis_collection.find({"user_id": ObjectId(user_id)})
#     #return list(records)

# -------- FEEDBACK --------
@app.post("/feedback")
def feedback(data: dict, token: str = ""):
    payload = jwt.decode(token, os.getenv("JWT_SECRET_KEY"), algorithms=["HS256"])

    feedback_collection.insert_one({
        "user_id": ObjectId(payload["sub"]),
        "analysis_id": ObjectId(data["analysis_id"]),
        "rating": data["rating"],
        "comments": data.get("comments")
    })

    return {"message": "Feedback saved"}

# -------- DELETE SINGLE HISTORY --------
@app.delete("/history/{analysis_id}")
def delete_history(analysis_id: str, token: str = ""):
    payload = jwt.decode(token, os.getenv("JWT_SECRET_KEY"), algorithms=["HS256"])
    user_id = payload["sub"]

    success = delete_single_history(user_id, analysis_id)
    if not success:
        raise HTTPException(status_code=404, detail="History not found")

    return {"message": "History deleted successfully"}

# -------- DELETE ALL HISTORY --------
@app.delete("/history")
def delete_all(token: str = ""):
    payload = jwt.decode(token, os.getenv("JWT_SECRET_KEY"), algorithms=["HS256"])
    user_id = payload["sub"]

    count = delete_all_history(user_id)
    return {
        "message": "All history deleted",
        "deleted_count": count
    }

# -------- RECOMMENDATIONS --------
def extract_concerns(result):
    concerns = []

    if result["acne"]["count"] > 0:
        concerns.append("acne")

    if result["pimples"]["count"] > 0:
        concerns.append("pimples")

    attrs = result["attributes"]
    if attrs["dryness"] > 60:
        concerns.append("dry")
    if attrs["redness"] > 60:
        concerns.append("sensitive")

    return list(set(concerns))

@app.get("/recommendations/{analysis_id}")
def recommendations(analysis_id: str, token: str = ""):
    payload = jwt.decode(token, os.getenv("JWT_SECRET_KEY"), algorithms=["HS256"])
    user_id = payload["sub"]

    analysis = analysis_collection.find_one({
        "_id": ObjectId(analysis_id),
        "user_id": ObjectId(user_id)
    })

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    result = analysis["result"]
    concerns = extract_concerns(result)

    products = list(products_collection.find({
        "concerns": {"$in": concerns}
    }))

    response_products = []
    for p in products:
        response_products.append({
            "name": p["name"],
            "brand": p["brand"],
            "image_url": p["image_url"],
            "price": p["price"],
            "buy_url": p["amazon_url"]
        })

    return {
        "based_on": concerns,
        "products": response_products
    }



# @app.get("/recommendations")
# def recommendations(analysis_id: str, token: str = ""):
#     payload = jwt.decode(token, os.getenv("JWT_SECRET_KEY"), algorithms=["HS256"])
#     user_id = payload["sub"]

#     analysis = analysis_collection.find_one({
#         "_id": ObjectId(analysis_id),
#         "user_id": ObjectId(user_id)
#     })

#     if not analysis:
#         raise HTTPException(status_code=404, detail="Analysis not found")

#     skin_type = analysis["result"]["skin_type"]
#     concern = analysis["result"]["concern"]

#     products = get_recommended_products(skin_type, concern)

#     return {
#         "skin_type": skin_type,
#         "concern": concern,
#         "products": products
#     }
