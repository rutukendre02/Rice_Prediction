import config
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image

class RiceClassifier():
    def __init__(self):
        self.model = None
        
    def load_model(self):
        self.model = tf.keras.models.load_model(config.MODEL_PATH)

    def preprocess_image(self, image_path):
        img = image.load_img(image_path, target_size=config.IMAGE_SIZE)
        img_array = image.img_to_array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    
    def predict(self, image_path):
        if self.model is None:
            self.load_model()
            
        test_image = self.preprocess_image(image_path)
        prediction = self.model.predict(test_image, verbose = 0)
        
        class_names = [
            "Arborio",
            "Basmati",
            "Ipsala",
            "Jasmine",
            "Karacadag"
        ]
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        return predicted_class