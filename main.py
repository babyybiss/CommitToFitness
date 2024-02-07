from fastapi import FastAPI, Request, Depends, Form, status
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import HTTPException
from fastapi.responses import StreamingResponse
from typing import List 

from pydantic import BaseModel
import src.squats.squats as sq
import cv2
import numpy as np
import io
import base64
from PIL import Image
from starlette.responses import RedirectResponse

import os

app = FastAPI()
abs_path = os.path.dirname(os.path.realpath("__file__"))
templates = Jinja2Templates(directory=f"{abs_path}/templates")
cap = cv2.VideoCapture(0)  # Use webcam (camera index 0) instead of a video file

# 정적 파일 제공 경로 설정
app.mount("/static", StaticFiles(directory="static"), name="static")

class SquatCounter:
    def __init__(self):
        self.count = 0
        self.dir = 1

    def reset_count(self):
        self.count = 0
        
squat_counter = SquatCounter()
        
@app.get("/")
def root(req: Request):
    return templates.TemplateResponse("index.html", {"request": req})

@app.get("/squats/counting")
def getCountingTemplate(req: Request):
    return templates.TemplateResponse("squats/sq_counting.html", {"request": req})

@app.get("/squats/posture")
def getPostureTemplate(req: Request):
    return templates.TemplateResponse("squats/sq_posture.html", {"request": req})


# Generator function to capture video frames, process them, and yield them
def squatsGen():
    cap = cv2.VideoCapture(0)  # 0 for the default webcam
    while True:
        success, img = cap.read()
        if not success:
            break
        # Apply squat detection logic to the frame
       
        frame = sq.sq_func(img, squat_counter)  # Adapt sq_func to process and return the frame
        print("frame? : ", frame)
        # Encode the frame in JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = buffer.tobytes()
        
        # Yield the frame in the multipart/x-mixed-replace format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg_as_text + b'\r\n\r\n')

@app.get("/squats/count_reset")
def squats_count_reset(req : Request):
    squat_counter.reset_count()
    return templates.TemplateResponse("squats/sq_counting.html", {"request": req})

@app.get('/squats/video_feed')
def video_feed():
    # Return the streaming response using the generator
    return StreamingResponse(squatsGen(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/curl")
def read_curl(req : Request):
    return templates.TemplateResponse("curl/index.html", {"request": req})