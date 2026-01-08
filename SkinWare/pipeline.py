import cv2
import numpy as np
import mediapipe as mp
from skimage import measure
#from mediapipe.python.solutions.face_mesh import FaceMesh


def is_blurry(gray):
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = 3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = 3)
    tenengrad = np.mean(sobelx ** 2 + sobely ** 2)

    print(f"Blur Metrics - Laplacian Variance: {lap_var:.2f}, Tenengrad: {tenengrad:.2f}")
    return lap_var < 25 and tenengrad < 20

def check_image_quality(image):
    reasons = []
    h, w, _ = image.shape
    if h < 224 or w < 224:
        reasons.append("Low resolution of Image. Please take the image again")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if is_blurry(gray):
        reasons.append("Image is blurry. Please take the image again")

    brightness = np.mean(gray)
    if brightness < 50:
        reasons.append("Image is too dark. Please take the image again")
    elif brightness > 200:
        reasons.append("Image is too bright. Please take the image again")
    
    return len(reasons) == 0, reasons

def normalize_lighting(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def relaxed_skin_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 30, 60], dtype=np.uint8)
    upper = np.array([25, 180, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

def is_skin_dominated_image(image, threshold = 0.6):
    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(
        ycbcr,
        np.array([0, 133, 77], np.uint8),
        np.array([255, 173, 127], np.uint8)
    )

    skin_ratio = cv2.countNonZero(skin_mask) / (image.shape[0] * image.shape[1])
    return skin_ratio > threshold

def face_skin_segmentation(image):
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode = True,
        max_num_faces=1,
        refine_landmarks=True
        )
    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None, None
    
    landmarks = result.multi_face_landmarks[0]

    FACE_OVAL = [
        10,338,297,332,284,251,389,356,454,323,361,288,
        397,365,379,378,400,377,152,148,176,149,150,
        136,172,58,132,93,234,127,162,21,54,103,67,109
    ]
    LEFT_EYE = [33,133,160,159,158,157,173]
    RIGHT_EYE = [362,263,387,386,385,384,398]
    LIPS = [61,146,91,181,84,17,314,405,321,375,291,308]
    LEFT_EYEBROW = [70,63,105,66,107]
    RIGHT_EYEBROW = [336,296,334,293,300]
    NOSE = [1,2,98,327,168,197,5]

    face_mask = np.zeros((h, w), np.uint8)
    eye_mask = np.zeros((h, w), np.uint8)
    lip_mask = np.zeros((h, w), np.uint8)
    eyebrow_mask = np.zeros((h, w), np.uint8)
    nose_mask = np.zeros((h, w), np.uint8)

    def draw(indices, mask):
        pts = np.array(
            [(int(landmarks.landmark[i].x * w),
              int(landmarks.landmark[i].y * h)) for i in indices],
              np.int32
        )
        cv2.fillPoly(mask, [pts], 255)
    
    draw(FACE_OVAL, face_mask)
    draw(LEFT_EYE, eye_mask)
    draw(RIGHT_EYE, eye_mask)
    draw(LEFT_EYEBROW, eyebrow_mask)
    draw(RIGHT_EYEBROW, eyebrow_mask)
    draw(LIPS, lip_mask)
    draw(NOSE, nose_mask)

    # Expand nose region
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    nose_mask = cv2.dilate(nose_mask, kernel, iterations=1)

    skin_mask = face_mask.copy()
    skin_mask = cv2.subtract(skin_mask, eye_mask)
    skin_mask = cv2.subtract(skin_mask, eyebrow_mask)
    skin_mask = cv2.subtract(skin_mask, lip_mask)
    skin_mask = cv2.subtract(skin_mask, nose_mask)


    # Refining skin
    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    skin_color = cv2.inRange(
        ycbcr,
        np.array([0, 133, 77], np.uint8),
        np.array([255, 173, 127], np.uint8)
    )

    skin_mask = cv2.bitwise_and(skin_mask, skin_color)
    final_mask = skin_mask.copy()

    #Remove upper forehead since it have HAIRRR
    hairline_y = min([int(landmarks.landmark[i].y * h) for i in FACE_OVAL])
    hairline_cutoff = hairline_y + int(0.08 * h)
    final_mask[0 : hairline_cutoff, :] = 0

    # Fallback
    if cv2.countNonZero(final_mask) < 500:
        relaxed = relaxed_skin_mask(image)
        final_mask = cv2.bitwise_and(relaxed, face_mask)

    skin_only = cv2.bitwise_and(image, image, mask = final_mask)
    return final_mask, skin_only


def body_skin_segmentation(image):
    mp_selfie = mp.solutions.selfie_segmentation
    selfie = mp_selfie.SelfieSegmentation(model_selection = 1)
    
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    seg = selfie.process(rgb)
    person_mask = (seg.segmentation_mask > 0.5).astype(np.uint8) * 255

    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    skin_color = cv2.inRange(
        ycbcr,
        np.array([0, 133, 77], np.uint8),
        np.array([255, 173, 127], np.uint8)
    )

    skin_mask = cv2.bitwise_and(person_mask, skin_color)
    if cv2.countNonZero(skin_mask) < 1000:
        relaxed = relaxed_skin_mask(image)
        skin_mask = cv2.bitwise_and(person_mask, relaxed)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

    labels = measure.label(skin_mask)
    regions = measure.regionprops(labels)
    if not regions:
        return None, None
    
    largest = max(regions, key = lambda r: r.area)
    final_mask = (labels == largest.label).astype(np.uint8) * 255

    cv2.imwrite("debug_body_mask.png", final_mask)
    skin_only = cv2.bitwise_and(image, image, mask = final_mask)

    return final_mask, skin_only

def run_full_pipeline(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise Exception(f"Could not load image from path: {image_path}")

    ok, reasons = check_image_quality(image)
    if not ok:
        raise Exception(f"Image rejected. Why?\n{reasons}")

    image = normalize_lighting(image)

    if is_skin_dominated_image(image):
        print("Skin-dominated image detected so using full image")
        mask, skin = face_skin_segmentation(image)

        if mask is not None and cv2.countNonZero(mask) > 500:
            print("Skin dominated image, removing face landmarks like: eyes, lips, nose")
            return skin, mask, "skin"

        return image, np.ones(image.shape[:2], dtype=np.uint8) * 255, "skin"

    # Face pipeline
    mask, skin = face_skin_segmentation(image)
    if mask is not None and cv2.countNonZero(mask) > 500:
        print("Face skin detected")
        return skin, mask, "face"

    # Body pipeline
    mask, skin = body_skin_segmentation(image)
    if mask is not None and cv2.countNonZero(mask) > 1000:
        print("Body skin detected")
        return skin, mask, "body"

    raise Exception("No skin detected in the image. Please take the image again")
