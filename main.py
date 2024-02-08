from fastapi import FastAPI, Request, Depends, Form, status, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
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
import mediapipe as mp
import json
import math

app = FastAPI()
abs_path = os.path.dirname(os.path.realpath("__file__"))
templates = Jinja2Templates(directory=f"{abs_path}/templates")
cap = cv2.VideoCapture(0)  # Use webcam (camera index 0) instead of a video file


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

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



# 프레임과 함께 전송할 데이터 초기화
wrist_distance = 0
shoulder_distance = 0
w_s_ratio = 0

@app.get("/pushup", response_class=HTMLResponse)
async def read_root():
    with open("templates/pushup/pushup.html", "r", encoding='utf-8') as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.websocket("/wsPushup")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            frame = base64.b64decode(data.split(',')[1])
            nparr = np.frombuffer(frame, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            encoded_frame = process_frame(img_np)

            result_data = json.dumps({"wrist_distance": wrist_distance,"shoulder_distance": shoulder_distance,"w_s_ratio": w_s_ratio, "good_pushup" : good_pushup(w_s_ratio),"frame": encoded_frame})
            await websocket.send_text(result_data)
    except WebSocketDisconnect:
        print("WebSocket connection closed")
    except Exception as e:
        print(f"Error: {e}")

def calculate_wrist_distance(landmarks):
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    distance = math.sqrt((left_wrist.x - right_wrist.x) ** 2 + (left_wrist.y - right_wrist.y) ** 2)
    return distance

def calculate_shoulder_distance(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    distance = math.sqrt((left_shoulder.x - right_shoulder.x) ** 2 + (left_shoulder.y - right_shoulder.y) ** 2)
    return distance

def encode_frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def process_frame(frame):

    # 프레임과 함께 전송할 데이터를 담을 전역변수 선언
    global wrist_distance
    global shoulder_distance
    global w_s_ratio
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style())
        
        # 모델 동작 시점의 랜드마크로 관절 사이 거리 계산
        wrist_distance = calculate_wrist_distance(results.pose_landmarks.landmark)
        shoulder_distance = calculate_shoulder_distance(results.pose_landmarks.landmark)
        w_s_ratio = wrist_distance / shoulder_distance
    # 인코딩된 프레임을 반환
    return encode_frame_to_base64(frame)

def good_pushup(w_s_ratio):
    if w_s_ratio < 1.3:
        result = '삼두근에 좋아요'
    elif w_s_ratio >= 1.3 and w_s_ratio < 1.7:
        result = '가슴 운동에 좋아요'
    else:
        result = '어깨를 다칠수도 있어요'
    return result

# 프레임과 함께 전송할 데이터 초기화
elbow_angle = 0
e_a_success = ""



@app.get("/shoulderPress", response_class=HTMLResponse)
async def read_root():
    with open("templates/shoulderPress/shoulderPress.html", "r", encoding='utf-8') as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.websocket("/wsShoulderPress")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            frame = base64.b64decode(data.split(',')[1])
            nparr = np.frombuffer(frame, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            encoded_frame = shoulderPress_process_frame(img_np)

            result_data = json.dumps({"elbow_angle": elbow_angle, "e_a_success": e_a_success, "frame": encoded_frame})
            await websocket.send_text(result_data)
    except WebSocketDisconnect:
        print("WebSocket connection closed")
    except Exception as e:
        print(f"Error: {e}")

'''
def calculate_wrist_distance(landmarks):
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    distance = math.sqrt((left_wrist.x - right_wrist.x) ** 2 + (left_wrist.y - right_wrist.y) ** 2)
    return distance

def calculate_shoulder_distance(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    distance = math.sqrt((left_shoulder.x - right_shoulder.x) ** 2 + (left_shoulder.y - right_shoulder.y) ** 2)
    return distance
'''

# 세 점 사이의 각도를 계산하는 함수입니다.
def calculate_angle(a, b, c):
    # 세 점 (a, b, c)의 좌표를 사용하여 각도 계산
    a = np.array(a) # 첫 번째 점
    b = np.array(b) # 두 번째 점 (관절점)
    c = np.array(c) # 세 번째 점

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def shoulderPress_process_frame(frame):
    global elbow_angle
    global e_a_success

    # 프레임을 RGB로 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        # 포즈 랜드마크 그리기
        mp.solutions.drawing_utils.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style())

        # 오른팔 각도 계산
        landmarks = results.pose_landmarks.landmark
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        angle = calculate_angle(shoulder, elbow, wrist)
        elbow_angle = angle
        # 계산된 각도에 따라 색상을 결정합니다.
        if 80 <= angle <= 100:
            color = (0, 255, 0)  # 특정 각도 범위일 때 녹색
            e_a_success = "팔을 올리세요!!"
        elif 160 <= angle <= 190:
            color = (0, 255, 0)  # 특정 각도 범위일 때 녹색
            e_a_success = "팔을 내리세요!!"
        else:
            color = (255, 255, 255)  # 그 외는 흰색

        # 새 색상으로 랜드마크와 연결선을 다시 그립니다.
        landmark_color = mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
        connection_color = mp_drawing.DrawingSpec(color=color, thickness=2)
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, 
            [(mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value),
            (mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_WRIST.value)],
            landmark_drawing_spec=landmark_color,
            connection_drawing_spec=connection_color)

        # 각도를 프레임에 덧그리기
        cv2.putText(frame, str(int(angle)), 
                    tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
    # 여기서 encode_frame_to_base64 함수로 인코딩된 프레임을 반환해야 하지만,
    # 이 예제에서는 그냥 frame을 반환합니다. 실제 사용 시 적절한 인코딩 함수를 사용하세요.
    return encode_frame_to_base64(frame)