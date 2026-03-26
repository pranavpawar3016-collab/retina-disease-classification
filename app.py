import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# LOAD MODEL
model = load_model("cnn.keras", compile=False)

CLASS_NAMES = [
    'Cataract',
    'Diabetes',
    'Glaucoma',
    'Normal',
    'Other'
]

# ROUTES
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

# PREDICTION
@app.route("/output", methods=["POST"])
def output():

    file = request.files["image"]

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    img = load_img(filepath, target_size=(224,224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    preds = model.predict(img, verbose=0)[0]
    print("Predictions:", preds)

    idx = np.argmax(preds)
    prediction = CLASS_NAMES[idx]
    confidence = float(preds[idx]) * 100

    return render_template(
        "result.html",
        prediction=prediction,
        confidence=round(confidence,2),
        image_path=filepath
    )

if __name__ == "__main__":
    import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
