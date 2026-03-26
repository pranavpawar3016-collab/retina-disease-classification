import os
import numpy as np
import gdown
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)

# =========================
# FOLDER SETUP
# =========================
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =========================
# MODEL DOWNLOAD (Google Drive)
# =========================
FILE_ID = "1HiqFT-7VpOXwO0Tu37sVgFdm0-fYbg17"
url = f"https://drive.google.com/uc?id={FILE_ID}"
model_path = "cnn.keras"

# Download model if not present
model = None

# =========================
# CLASS NAMES
# =========================
CLASS_NAMES = [
    'Cataract',
    'Diabetes',
    'Glaucoma',
    'Normal',
    'Other'
]

# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")

# =========================
# PREDICTION
# =========================
@app.route("/output", methods=["POST"])
def output():

    file = request.files["image"]

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    img = load_img(filepath, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    global model
    
    if model is None:
        FILE_ID = "1HiqFT-7VpOXwO0Tu37sVgFdm0-fYbg17"
        url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
        model_path = "cnn.keras"
    
        if not os.path.exists(model_path):
            print("Downloading model...")
            gdown.download(url, model_path, quiet=False)
    
        print("Loading model...")
        model = load_model(model_path, compile=False)
    
    preds = model.predict(img, verbose=0)[0]
    print("Predictions:", preds)

    idx = np.argmax(preds)
    prediction = CLASS_NAMES[idx]
    confidence = float(preds[idx]) * 100

    return render_template(
        "result.html",
        prediction=prediction,
        confidence=round(confidence, 2),
        image_path=filepath
    )

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
