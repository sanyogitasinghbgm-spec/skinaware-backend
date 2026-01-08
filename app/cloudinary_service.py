import cloudinary.uploader
from app.cloudinary_config import cloudinary

def upload_image(file):
    result = cloudinary.uploader.upload(
        file,
        folder="skinaware"
    )
    return result["secure_url"]
