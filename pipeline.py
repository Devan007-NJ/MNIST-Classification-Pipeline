import numpy as np
import joblib
from PIL import Image

class DigitalPipeline:
    def __init__(self,model_path):
        self.model=joblib.load(model_path)
        pass
    def preprocess(self,image_path):
        img = Image.open(image_path).convert("L")
        img = img.resize((28, 28)) 
        img = np.array(img, dtype=np.float32)
        if img.mean() > 127:
            img = 255 - img
        img /= 255.0
        return img.flatten().reshape(1, -1)
    def predict(self,image_path):
        X=self.preprocess(image_path)
        return int(self.model.predict(X))

if __name__=='__main__':
    pipeline=DigitalPipeline("model.pkl")
    digit=pipeline.predict("digit.png")
    print("Predicted digit:",digit)

