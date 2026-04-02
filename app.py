from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load model
model = load_model("astronomy_image_classifier.h5")

# Automatically read folder names
categories = os.listdir("space images")

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    if file:
        filepath = os.path.join("static", file.filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(128,128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)
        predicted_class = categories[np.argmax(prediction)]

        return render_template(
            'index.html',
            prediction=predicted_class.upper(),
            img_path=filepath
        )

    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
