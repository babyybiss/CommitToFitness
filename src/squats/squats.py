
import cv2
import numpy as np
import time
from src.squats.models.PoseModule import poseDetector
#from models.PoseModule import poseDetector as detector

detector = poseDetector()

# STEP 1:
#detector = pm.poseDetector()
cap = cv2.VideoCapture("vid/squat2.mp4") 


def sq_func(img, squat_counter):
    squat_counter.count
    squat_counter.dir

    print("1111 mg? : ", img)
    img = detector.findPose(img, draw=False)
    print("2222 img? : ", img)
    #print(lmList)
    lmList = detector.findPosition(img, draw =False)
    
    # STEP 4: find angle
    if len(lmList) != 0:
        #right leg
        right_angle = detector.findAngle(img, 24, 26, 28, draw=True)
        # left leg
        left_angle = detector.findAngle(img, 23, 25, 27, draw=True)
        
        right_waist = detector.findAngle(img, 12, 24, 26, draw=True)
        left_waist = detector.findAngle(img, 11, 23, 25, draw=True)
        print("\n waist angle \n", left_waist, right_waist,"\n")
        
        
        # manipulate angle
        right_per = np.interp(right_angle,(120,170),(0,100))
        left_per = np.interp(left_angle,(120,170),(0,100))
        bar = np.interp(right_angle + left_angle, (220,310), (650,100))
        
        print(right_angle, right_per)
        print(left_angle, left_per)
        
        # check for the squats
        if right_per == 100 and left_per == 100:
            if squat_counter.dir == 0:
                squat_counter.count += 0.5
                squat_counter.dir = 1
                print(right_per, left_per)
        if right_per == 0 and left_per == 0:
            if squat_counter.dir == 1:
                squat_counter.count += 0.5
                squat_counter.dir = 0
                print(right_per, left_per)
                
        # Calculate the average percentage
        combined_per = (right_per + left_per) / 2

        cv2.rectangle(img, (1100, 100), (1175, 650), (0,255,0), 3)
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), (255,255,255), cv2.FILLED)
        cv2.putText(img, f"{int(combined_per)} %", (1080, 60), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 255, 255), 3)
        cv2.putText(img, f"Counts: {str(int(squat_counter.count))}", (50,100), cv2.FONT_HERSHEY_PLAIN,5,
                                        (255,255,255),5)
    return img
    #cv2.imshow("Image", img)
    #cv2.waitKey(1)
    
    
