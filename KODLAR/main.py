import base64
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from services.labels import WORD_LABEL_MAP, WORD_LABELS
from services.mediapipe_service import HandSignDetector
from services.model_service import WordModelPredictor


BASE_DIR = Path(__file__).parent
FRONTEND_PATH = BASE_DIR / "frontend" / "index.html"
DATA_DIR = BASE_DIR / "data"
WORD_DATASET_PATH = DATA_DIR / "word_landmarks.csv"
WORD_MODEL_PATH = BASE_DIR / "models" / "word_model.pkl"

detector = HandSignDetector()
word_predictor = WordModelPredictor(WORD_MODEL_PATH)
app = FastAPI()


@app.get("/")
async def root():
    html_content = FRONTEND_PATH.read_text(encoding="utf-8")
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/labels/words")
async def get_word_labels():
    return WORD_LABELS


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    mode = "word"
    collection_label = None
    saved_samples = 0

    try:
        while True:
            data = await websocket.receive_text()

            if data.startswith("{"):
                command = json.loads(data)
                mode, collection_label, saved_samples = handle_command(
                    command,
                    mode,
                    collection_label,
                    saved_samples,
                )
                await websocket.send_json(
                    {
                        "status": "info",
                        "mode": mode,
                        "collection_label": collection_label,
                        "saved_samples": saved_samples,
                    }
                )
                continue

            if data.startswith("data:image"):
                frame = decode_frame(data)

                if frame is None:
                    await websocket.send_json({"error": "Görüntü çözülemedi"})
                    continue

                result = detector.process_frame(frame)
                features = result.pop("features", None)
                result["mode"] = mode

                if mode == "word" and collection_label and features:
                    append_word_sample(collection_label, features)
                    saved_samples += 1
                    result["collection_label"] = collection_label
                    result["saved_samples"] = saved_samples
                elif mode == "word" and features:
                    prediction = word_predictor.predict(features)

                    if prediction:
                        result["status"] = "success"
                        result["prediction"] = prediction
                        result["message"] = (
                            f"{prediction['text']} "
                            f"(%{int(prediction['confidence'] * 100)})"
                        )

                if mode == "letter":
                    result["message"] = "Harf modu seçili. Harf modeli henüz eğitilmedi."

                await websocket.send_json(result)
                continue

            response = f"Arka yüz mesajı ('{data}') aldı"
            await websocket.send_text(response)

    except WebSocketDisconnect:
        print("İstemci bağlantıyı kopardı.")


def handle_command(command, mode, collection_label, saved_samples):
    command_type = command.get("type")

    if command_type == "set_mode":
        selected_mode = command.get("mode")
        if selected_mode in {"word", "letter"}:
            return selected_mode, None, 0

    if command_type == "start_collection":
        label = command.get("label")
        if label in WORD_LABEL_MAP:
            return mode, label, 0

    if command_type == "stop_collection":
        return mode, None, saved_samples

    return mode, collection_label, saved_samples


def decode_frame(data_url):
    try:
        _, base64_str = data_url.split(",", 1)
        img_bytes = base64.b64decode(base64_str)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except ValueError:
        return None


def append_word_sample(label, features):
    DATA_DIR.mkdir(exist_ok=True)
    file_exists = WORD_DATASET_PATH.exists()

    with WORD_DATASET_PATH.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        if not file_exists:
            writer.writerow(["label", *feature_columns()])

        writer.writerow([label, *features])


def feature_columns():
    columns = []

    for landmark_index in range(21):
        columns.extend(
            [
                f"x{landmark_index}",
                f"y{landmark_index}",
                f"z{landmark_index}",
            ]
        )

    return columns
