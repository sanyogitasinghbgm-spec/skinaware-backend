from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))
db = client["skinaware_db"]

users_collection = db["users"]
analysis_collection = db["analysis"]
feedback_collection = db["feedback"]
products_collection = db["products"]
