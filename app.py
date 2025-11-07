from flask import Flask, render_template, request, jsonify, send_from_directory, session
import os, io, json, base64, requests, joblib, bcrypt
from datetime import datetime
from pymongo import MongoClient
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -------------------------------------------------------
# Flask Configuration
# -------------------------------------------------------
app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------------------------------------
# Load ML Model + Encoder
# -------------------------------------------------------
MODEL_PATH = '/etc/secrets/best_model_1.keras'
ENCODER_PATH = '/etc/secrets/label_encoder_3.pkl'
model, label_encoder, CLASS_NAMES = None, None, None
IMG_SIZE = (128, 128)

try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
    else:
        print(f"⚠️ Model not found at {MODEL_PATH}")
    if os.path.exists(ENCODER_PATH):
        label_encoder = joblib.load(ENCODER_PATH)
        CLASS_NAMES = label_encoder.classes_
        print("✅ Label encoder loaded successfully.")
except Exception as e:
    print(f"❌ Error loading ML model: {e}")

# -------------------------------------------------------
# MongoDB Connection
# -------------------------------------------------------
try:
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise Exception("MONGO_URI not set in environment")
    client = MongoClient(mongo_uri)
    db = client["fabric_defect_db"]
    users_collection = db["users"]
    print("✅ Connected to MongoDB successfully via Render/Atlas!")
except Exception as e:
    print(f"❌ MongoDB connection error: {e}")
    users_collection = None

# -------------------------------------------------------
# Helper Functions
# -------------------------------------------------------
def preprocess_image(image_pil):
    image_pil = image_pil.resize(IMG_SIZE)
    arr = img_to_array(image_pil)
    if len(arr.shape) == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[2] == 4:
        arr = arr[..., :3]
    batch = np.expand_dims(arr, axis=0)
    return preprocess_input(batch)

def predict_defect(image_pil):
    if model is None or label_encoder is None:
        return None, None, None
    try:
        processed = preprocess_image(image_pil)
        preds = model.predict(processed, verbose=0)
        prob_good = float(preds[0][0])
        prob_defect = 1.0 - prob_good
        pred_class = "good" if prob_good > 0.5 else "defect"
        conf = round(max(prob_good, prob_defect) * 100, 2)
        all_preds = {"good": prob_good * 100, "defect": prob_defect * 100}
        return pred_class, conf, all_preds
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None, None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# -------------------------------------------------------
# Routes
# -------------------------------------------------------
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload (file or URL) and run prediction."""
    # --- Temporarily disable login check ---
    # if 'username' not in session:
    #     return jsonify({'success': False, 'message': 'Not authenticated'}), 401

    upload_type = request.form.get('upload_type')
    if upload_type == 'file':
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file provided'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'}), 400
        if file and allowed_file(file.filename):
            name, ext = os.path.splitext(secure_filename(file.filename))
            filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            data = file.read()
            file.seek(0)
            file.save(path)

            try:
                image_pil = Image.open(io.BytesIO(data))
                pred_class, conf, all_preds = predict_defect(image_pil)
                result = {
                    "class": pred_class,
                    "confidence": conf,
                    "is_defect": pred_class != "good",
                    "all_predictions": all_preds,
                }
            except Exception as e:
                print(f"Prediction error: {e}")
                result = None

            return jsonify({
                "success": True,
                "message": "File uploaded successfully",
                "filename": filename,
                "prediction": result
            })
        return jsonify({'success': False, 'message': 'Invalid file type'}), 400

    elif upload_type == 'url':
        image_url = request.form.get('image_url', '').strip()
        if not image_url:
            return jsonify({'success': False, 'message': 'No URL provided'}), 400
        try:
            resp = requests.get(image_url, timeout=10, stream=True)
            resp.raise_for_status()
            image_pil = Image.open(io.BytesIO(resp.content))
            pred_class, conf, all_preds = predict_defect(image_pil)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"url_{timestamp}.jpg"
            with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'wb') as f:
                f.write(resp.content)
            result = {
                "class": pred_class,
                "confidence": conf,
                "is_defect": pred_class != "good",
                "all_predictions": all_preds,
            }
            return jsonify({
                "success": True,
                "message": "Image downloaded successfully",
                "filename": filename,
                "prediction": result
            })
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 400

    return jsonify({'success': False, 'message': 'Invalid upload type'}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# -------------------------------------------------------
# JSON Error Handlers
# -------------------------------------------------------
@app.errorhandler(404)
def not_found_error(e):
    return jsonify({'error': 'Route not found', 'message': str(e)}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal Server Error', 'message': str(e)}), 500

# -------------------------------------------------------
# Run the App
# -------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
