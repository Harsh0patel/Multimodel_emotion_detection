from fastapi import APIRouter, HTTPException
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
    return {
        "Status" : 'OK',
        "Model_version" : MODEL_VERSION
    }