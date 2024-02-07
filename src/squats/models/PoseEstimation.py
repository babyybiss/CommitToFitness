#STEP 1 : import 
import cv2
import mediapipe as mp

#STEP7: create mp Draw obj
mpDraw = mp.solutions.drawing_utils

#STEP4: Detecting pose model
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# STEP2: Capture video
cap = cv2.VideoCapture('../vid/squat3.mp4')

# STEP3: Load video and show
while True:
    success, img = cap.read()
    
    #STEP5: change detected model BGR to RGB and detect from STEP 4
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    #print(results.pose_landmarks)

    #STEP6: Draw detected landmarks (connecting the pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    '''
    output
    landmark {
        x: 0.5377724
        y: 0.9554269
        z: -0.07881707
        visibility: 0.5625093
    '''
    
    #STEP 8: loops through the above outcome through landmark
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        print(id, lm)
        cx, cy = int(lm.x*w), int(lm.y*h)
        cv2.circle(img,(cx,cy), 10, (255,0,0), cv2.FILLED)
        
        
    cv2.imshow("Image", img)
    
    cv2.waitKey(1)
