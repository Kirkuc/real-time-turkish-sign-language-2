from pathlib import Path

try:
    import joblib
except ImportError:
    joblib = None


class WordModelPredictor:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model = None
        self.labels = {}
        self.load()

    def load(self):
        if joblib is None or not self.model_path.exists():
            return

        model_bundle = joblib.load(self.model_path)
        self.model = model_bundle["model"]
        self.labels = model_bundle.get("labels", {})

    def predict(self, features):
        if self.model is None:
            return None

        prediction = self.model.predict([features])[0]
        probabilities = self.model.predict_proba([features])[0]
        confidence = max(probabilities)
        label = self.labels.get(prediction, prediction)

        return {
            "label": prediction,
            "text": label,
            "confidence": round(float(confidence), 3),
        }
