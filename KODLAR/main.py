from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

app = FastAPI()

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
            data = await websocket.receive_text()

            cevap = f"Arka yüz mesajı ('{data}') aldı"
            await websocket.send_text(cevap)

    except WebSocketDisconnect:
        print("İstemci bağlantıyı kopardı.")