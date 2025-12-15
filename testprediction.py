from pipeline import DigitalPipeline
Pipeline=DigitalPipeline("model.joblib")
digit=Pipeline.predict("digit.png")
print("Predicted Digit:",digit)

