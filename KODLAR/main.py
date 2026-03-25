import base64
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from services.mediapipe_service import HandSignDetector

app = FastAPI()

# MediaPipe dedektörümüzü başlatıyoruz
detector = HandSignDetector()

@app.get("/")
async def root():
    with open("frontend/index.html", "r", encoding="utf-8") as dosya:
        html_icerik = dosya.read()

    return HTMLResponse(content=html_icerik, status_code=200)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            # İstemciden veriyi metin (Data URL/JSON) olarak alıyoruz
            data = await websocket.receive_text()

            # Eğer gelen veri "data:image" ile başlıyorsa (Kamera Frame'i)
            if data.startswith("data:image"):
                # "data:image/jpeg;base64," kısmını ayıkla
                header, base64_str = data.split(',', 1)
                
                # Base64 stringini byte'lara çevir
                img_bytes = base64.b64decode(base64_str)
                # Byte dizisini Numpy matrisine dönüştür
                np_arr = np.frombuffer(img_bytes, np.uint8)
                # Matrisi OpenCV BGR formatına çöz
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Görüntüyü MediaPipe ile işle
                    result = detector.process_frame(frame)
                    # Sonucu JSON olarak istemciye geri gönder
                    await websocket.send_json(result)
                else:
                    await websocket.send_json({"error": "Görüntü çözülemedi"})
            else:
                # Normal metin mesajları için eski davranış
                cevap = f"Arka yüz mesajı ('{data}') aldı"
                await websocket.send_text(cevap)

    except WebSocketDisconnect:
        print("İstemci bağlantıyı kopardı.")