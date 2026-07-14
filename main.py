from flask import Flask, render_template, request
import os
import config

from src.utils import RiceClassifier

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = config.UPLOAD_FOLDER

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok = True)

rice_classifier = RiceClassifier()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    if "rice_image" not in request.files:
        return "no file uploaded"
    
    file = request.files["rice_image"]
    
    if file.filename == "":
        return "no file selected"
    
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(image_path)
    
    predicted_class = rice_classifier.predict(image_path)
    return render_template("index.html", 
                           prediction = predicted_class,
                           image = file.filename)
    
if __name__ == "__main__":
    app.run(debug = True)