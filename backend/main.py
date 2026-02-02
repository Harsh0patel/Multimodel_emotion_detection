from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Multimodal Emotion Detection API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from routes import home_page_route, web_socket_route
app.include_router(home_page_route.router)
app.include_router(web_socket_route.router)

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)