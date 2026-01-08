from bson import ObjectId
from app.database import users_collection, analysis_collection, feedback_collection
from app.auth import hash_password, verify_password
from datetime import datetime
from app.database import products_collection

# ---------- USERS ----------
def create_user(user):
    user["password"] = hash_password(user["password"])
    users_collection.insert_one(user)
    return {"message": "User created"}

def authenticate_user(email, password):
    user = users_collection.find_one({"email": email})
    if user and verify_password(password, user["password"]):
        return user
    return None

# ---------- ANALYSIS ----------
def save_analysis(user_id, result, image_url=None):
    doc = {
        "user_id": ObjectId(user_id),
        "result": result,
        "image_url": image_url,
        "created_at": datetime.utcnow()
    }
    inserted = analysis_collection.insert_one(doc)
    analysis_id = str(inserted.inserted_id)
    #return doc
    return analysis_id 


def get_history(user_id):
    records = analysis_collection.find({"user_id": ObjectId(user_id)})
    history = []
    for record in records:
        record["_id"] = str(record["_id"])
        record["user_id"] = str(record["user_id"])

        skin_type = record["result"]["skin_type"]
        concern = record["result"]["concern"]

        products = get_recommended_products(skin_type, concern)
        record["recommended_products"] = products

        history.append(record)

    return history

# ---------- FEEDBACK ----------
def save_feedback(user_id, analysis_id, rating, comments=None):
    feedback_collection.insert_one({
        "user_id": ObjectId(user_id),
        "analysis_id": ObjectId(analysis_id),
        "rating": rating,
        "comments": comments
    })
    return {"message": "Feedback saved"}

# ---------- DELETE HISTORY ----------
def delete_single_history(user_id, analysis_id):
    result = analysis_collection.delete_one({
        "_id": ObjectId(analysis_id),
        "user_id": ObjectId(user_id)
    })

    if result.deleted_count == 0:
        return False
    return True


def delete_all_history(user_id):
    result = analysis_collection.delete_many({
        "user_id": ObjectId(user_id)
    })
    return result.deleted_count

# ---------- PRODUCTS ----------

def get_recommended_products(skin_type, concern):
    products = products_collection.find({
        "skin_type": skin_type,
        "concern": concern
    })

    result = []
    for p in products:
        p["_id"] = str(p["_id"])
        result.append(p)

    return result
