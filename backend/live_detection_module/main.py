from fastapi import FastAPI
from routes import home_page, live_detection

app = FastAPI()

app.include_router(home_page.router, prefix = "/")
app.include_router(live_detection.router, prefix = "/connect")