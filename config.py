import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR,
                          "artifacts",
                          "rice_classification.keras")

UPLOAD_FOLDER = os.path.join(BASE_DIR,
                             "static",
                             "uploads")
IMAGE_SIZE = (64,64)


                          