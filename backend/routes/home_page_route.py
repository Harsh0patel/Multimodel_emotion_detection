from fastapi import APIRouter
from fastapi.responses import JSONResponse
from configs import config

router = APIRouter()
MODEL_VERSION = config.MODEL_VERSION

@router.get('/')
def home_page():
    return JSONResponse(
        {"Message": "This is Multimodel mood detection fastapi endpoint."},
        status_code=200)

@router.get('/health')
def health():
    return JSONResponse(
        {"Status" : 'OK',
        "Model_version" : MODEL_VERSION},
        status_code=200
    )