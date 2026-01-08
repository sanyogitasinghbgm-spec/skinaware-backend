import numpy as np
from bson import ObjectId

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
    elif isinstance(obj, ObjectId):
        return str(obj)
    else:
        return obj
