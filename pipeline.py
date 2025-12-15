import numpy as np
from PIL import Image
try:
    class DigitalPipeline:
        def __init__(self, model_path):
            # Load model with explicit error handling
            import joblib
            self.model = joblib.load(model_path)
        
        def preprocess(self, image_path):
            img = Image.open(image_path).convert("L")
            
            # Increase contrast
            img = img.point(lambda x: 255 if x > 150 else 0)
            
            # Invert if background is white
            if np.array(img).mean() > 127:
                img = Image.fromarray(255 - np.array(img))
            
            # Get bounding box of digit
            bbox = img.getbbox()
            img = img.crop(bbox)
            
            # Resize while preserving aspect ratio
            img = img.resize((20, 20), Image.Resampling.LANCZOS)
            
            # Create 28x28 black canvas
            canvas = Image.new("L", (28, 28), 0)
            canvas.paste(img, ((28 - 20) // 2, (28 - 20) // 2))
            
            img = np.array(canvas, dtype=np.float32)
            img /= 255.0
            return img.flatten().reshape(1, -1)
        
        def predict(self, image_path):
            X = self.preprocess(image_path)
            return int(self.model.predict(X)[0])
except UnicodeDecodeError:
    pass
