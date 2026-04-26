from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "word_landmarks.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "word_model.pkl"

sys.path.append(str(BASE_DIR))

from services.labels import WORD_LABEL_MAP


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Veri seti bulunamadı: {DATASET_PATH}")

    dataset = pd.read_csv(DATASET_PATH)

    if dataset["label"].nunique() < 2:
        raise ValueError("Model eğitmek için en az 2 farklı etiket gerekir.")

    x = dataset.drop(columns=["label"])
    y = dataset["label"]

    stratify = y if y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred, zero_division=0))

    labels = {
        label_id: label_info["text"]
        for label_id, label_info in WORD_LABEL_MAP.items()
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "labels": labels}, MODEL_PATH)
    print(f"Model kaydedildi: {MODEL_PATH}")


if __name__ == "__main__":
    main()
