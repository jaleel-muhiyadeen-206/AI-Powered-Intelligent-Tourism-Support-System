# -*- coding: utf-8 -*-
"""Landmark Classifier App - For Streamlit Cloud Deployment"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import json
import tempfile

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
MODEL_PATH = 'FineTuned01-EfficientNetB0_CNN_Model.h5'
CLASS_NAMES_PATH = 'class_names.json'
CONFIDENCE_THRESHOLD = 0.65

# -----------------------------------------------------------------------------
# LOCATION MAPPING
# -----------------------------------------------------------------------------
LOCATION_MAP = {
    "Adams Peak": "Rathnapura, Sabaragamuwa Province, Sri Lanka",
    "Ancient City of Polonnaruwa": "Polonnaruwa, North Central Province, Sri Lanka",
    "Beruwala Light House": "Beruwala, Western Province, Sri Lanka",
    "British War Cemetery": "Kandy, Central Province, Sri Lanka",
    "Bundala National Park": "Hambantota, Southern Province, Sri Lanka",
    "Delft Island": "Jaffna, Northern Province, Sri Lanka",
    "Dowa Rock Temple": "Bandarawela, Uva Province, Sri Lanka",
    "Ganagaramaya Temple": "Colombo, Western Province, Sri Lanka",
    "Henarathgoda Botanical Gard": "Gampaha, Western Province, Sri Lanka",
    "Hortains Plain": "Nuwara Eliya, Central Province, Sri Lanka",
    "Independance Square": "Colombo, Western Province, Sri Lanka",
    "Jaya Sri Maha Bodhi": "Anuradhapura, North Central Province, Sri Lanka",
    "Lotus Tower": "Colombo, Western Province, Sri Lanka",
    "Maligawa Buddha Statue": "Kandy, Central Province, Sri Lanka",
    "Nine Arches Bridge": "Ella, Uva Province, Sri Lanka",
    "Pinnawala Elephant Orphanage": "Kegalle, Sabaragamuwa Province, Sri Lanka",
    "Sigiriya": "Matale, Central Province, Sri Lanka",
    "Sinharaja Forest": "Ratnapura, Sabaragamuwa Province, Sri Lanka",
    "Sri Dalada Maligawa": "Kandy, Central Province, Sri Lanka",
    "Star Fort": "Matara, Southern Province, Sri Lanka",
    "Turtle Hatchery": "Kosgoda, Southern Province, Sri Lanka",
    "Vavuniya Archaeological Museum": "Vavuniya, Northern Province, Sri Lanka",
    "Wilpattu National Park": "Puttalam, North Western Province, Sri Lanka",
    "Yapahuwa Rock Fortress": "Yapahuwa, North Western Province, Sri Lanka",
}

# -----------------------------------------------------------------------------
# GLOBAL VARIABLES
# -----------------------------------------------------------------------------
ocr_reader = None
face_cascade = None


def get_ocr_reader():
    global ocr_reader
    if ocr_reader is None:
        try:
            import easyocr
            ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        except Exception as e:
            print(f"OCR Load Error: {e}")
            return None
    return ocr_reader


def get_face_cascade():
    global face_cascade
    if face_cascade is None:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade


# Detects real humans, not statues. More stricter conditions used
def detect_humans(img_array):
    try:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        # Use stricter parameters - higher minSize to avoid detecting statues
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=6,
            minSize=(100, 100)
        )
        return len(faces) > 0
    except:
        return False


# Enhances blurry images for better detection for ocr_reader
def enhance_image_for_ocr(img_pil):
    enhancer = ImageEnhance.Sharpness(img_pil)
    img_enhanced = enhancer.enhance(2.5)
    enhancer = ImageEnhance.Contrast(img_enhanced)
    img_enhanced = enhancer.enhance(2.0)
    enhancer = ImageEnhance.Brightness(img_enhanced)
    img_enhanced = enhancer.enhance(1.2)
    return img_enhanced


# More flexible method for analyzing text on images for detecting landmarks
def flexible_ocr_match(detected_text, landmark_name):
    if not detected_text:
        return False

    detected_text = detected_text.lower()
    landmark_lower = landmark_name.lower()
    landmark_words = landmark_lower.split()

    # Check for direct match
    if landmark_lower in detected_text:
        return True

    # Get significant words (longer than 2 chars)
    significant_words = [word for word in landmark_words if len(word) > 2]

    if not significant_words:
        return False

    # Check if ANY significant word appears
    for word in significant_words:
        if word in detected_text:
            return True

    # Check partial matches
    detected_words = detected_text.split()
    for kw in significant_words:
        for dw in detected_words:
            if kw in dw or dw in kw:
                return True

    return False


# -----------------------------------------------------------------------------
# CLASSIFIER CLASS
# -----------------------------------------------------------------------------
class LandmarkClassifier:
    def __init__(self, model_path=MODEL_PATH, classes_path=CLASS_NAMES_PATH):
        self.model_path = model_path
        self.classes_path = classes_path
        self.model = None
        self.class_names = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        if not os.path.exists(self.classes_path):
            raise FileNotFoundError(f"Class names file not found at {self.classes_path}")

        self.model = tf.keras.models.load_model(self.model_path, safe_mode=False, compile=False)
        with open(self.classes_path, 'r') as f:
            self.class_names = json.load(f)

    def predict(self, image_input):
        try:
            # Load Image
            if isinstance(image_input, str):
                img = Image.open(image_input).convert('RGB')
            elif isinstance(image_input, Image.Image):
                img = image_input.convert('RGB')
            else:
                raise ValueError("Input must be file path or PIL Image")

            img_array = np.array(img)

            # 1. Human Validation - but be less strict
            if detect_humans(img_array):
                print("⚠️ Faces detected, but continuing anyway...")
                # Don't reject - just continue (statues might be detected)

            # 2. CNN Prediction
            img_resized = cv2.resize(img_array, (290, 290))
            img_normalized = img_resized / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)

            probs = self.model.predict(img_batch, verbose=0)[0]
            pred_idx = np.argmax(probs)
            confidence = float(np.max(probs))

            print(f"📊 CNN Confidence: {confidence:.2f}, Prediction: {self.class_names[pred_idx]}")

            # 3. Confidence Validation
            if confidence < CONFIDENCE_THRESHOLD:
                print(f"❌ Low confidence: {confidence:.2f} < {CONFIDENCE_THRESHOLD}")
                return None

            landmark = self.class_names[pred_idx].strip()
            location = LOCATION_MAP.get(landmark, "Unknown Location")

            # 4. OCR Verification
            if confidence < 0.90:
                ocr = get_ocr_reader()
                if ocr:
                    try:
                        # Try enhanced image first
                        enhanced_img = enhance_image_for_ocr(img)
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                        enhanced_img.save(temp_file.name)

                        ocr_results = ocr.readtext(
                            temp_file.name,
                            detail=0,
                            min_size=10,
                            contrast_ths=0.02,
                            text_threshold=0.2
                        )
                        detected_text = " ".join(ocr_results)
                        os.unlink(temp_file.name)

                        print(f"🔍 OCR detected: '{detected_text}'")

                        if flexible_ocr_match(detected_text, landmark):
                            print("✅ OCR Match found!")
                        else:
                            # Try original image
                            temp_file2 = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                            img.save(temp_file2.name)

                            ocr_results2 = ocr.readtext(
                                temp_file2.name,
                                detail=0,
                                min_size=10,
                                contrast_ths=0.02
                            )
                            detected_text2 = " ".join(ocr_results2)
                            os.unlink(temp_file2.name)

                            print(f"🔍 OCR (original): '{detected_text2}'")

                            if not flexible_ocr_match(detected_text2, landmark):
                                if confidence < 0.75:
                                    print("❌ No OCR match + low confidence")
                                    return None
                                else:
                                    print("⚠️ No OCR match, but high confidence - accepting")

                    except Exception as e:
                        print(f"OCR Error: {e}")
                        if confidence < 0.75:
                            return None
                else:
                    if confidence < 0.75:
                        return None

            return {
                'name': landmark,
                'place': location
            }

        except Exception as e:
            print(f"Prediction Error: {e}")
            return None


# -----------------------------------------------------------------------------
# INITIALIZATION
# -----------------------------------------------------------------------------
classifier = None


def init_classifier(model_path=MODEL_PATH, classes_path=CLASS_NAMES_PATH):
    global classifier
    classifier = LandmarkClassifier(model_path, classes_path)
    return classifier


def get_prediction(image_input):
    global classifier
    if classifier is None:
        classifier = LandmarkClassifier()
    return classifier.predict(image_input)