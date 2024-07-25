import cv2
import mediapipe as mp
import math
import numpy as np

#STEP 3: create a class that has objects and methods to detect the pose and find all the points(landmarks)
class poseDetector():
    
    def __init__(self, mode=False, complexity=1, smooth_landmarks=True, enable_seg=False, smooth_seg=True, detectionCon=0.5, trackCon=0.5):
        
        self.mode = mode
        self.complexity = complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_seg = enable_seg
        self.smooth_seg = smooth_seg
        self.detectionCon = detectionCon
        self.trackCon = trackCon
            
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smooth_landmarks, self.enable_seg, self.smooth_seg ,self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        img = cv2.convertScaleAbs(img)
        print("Image depth (before conversion):", img.dtype)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print("\n 3333 img : \n", img)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            print("\n 4444 img \n: ", img)
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, 
                                          self.mpPose.POSE_CONNECTIONS)
            print("\n 5555 img \n: ", img)
            return img
            '''
                output
                landmark {
                    x: 0.5377724
                    y: 0.9554269
                    z: -0.07881707
                    visibility: 0.5625093
            '''
        return img
    '''
    def findPosition(self, img, draw=True):
        self.lmList = []
        
        if self.results.pose_landmarks:
            print(self.results.pose_landmarks.landmark)
            print(type(self.results.pose_landmarks.landmark))
        
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                print("test")
                h, w, c = img.shape
                #print(id, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id, cx, cy])
                if draw: 
                    cv2.circle(img,(cx,cy), 10, (255,0,0), cv2.FILLED)
        return self.lmList
    '''
    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), lm.z
                self.lmList.append([id, cx, cy, cz])  # Including z value
                if draw:
                    # Here, you could adjust the circle size or color based on the z value
                    # This example keeps it simple and uses a fixed size and color
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        return self.lmList
    
    def findAngle(self, img, p1, p2, p3, draw=True):
        # Retrieve coordinates including z values
        _, x1, y1, z1 = self.lmList[p1]
        _, x2, y2, z2 = self.lmList[p2]
        _, x3, y3, z3 = self.lmList[p3]

        # Create vectors from points: v1 is from p2 to p1, v2 is from p2 to p3
        v1 = np.array([x1 - x2, y1 - y2, z1 - z2])
        v2 = np.array([x3 - x2, y3 - y2, z3 - z2])

        # Calculate the angle using the dot product and magnitude of vectors
        dot_product = np.dot(v1, v2)
        magnitude_v1 = np.linalg.norm(v1)
        magnitude_v2 = np.linalg.norm(v2)
        angle = math.degrees(math.acos(dot_product / (magnitude_v1 * magnitude_v2)))

        # Draw
        if draw:
            # 2D drawing part remains the same, as we're visualizing on a 2D plane
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            
            # put angle text
            cv2.putText(img, str(int(angle)), (x2 - 20, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        return angle    
'''
    def findAngle(self, img, p1, p2, p3, draw = True):
        _, x1, y1 = self.lmList[p1] 
        _, x2, y2 = self.lmList[p2] 
        _, x3, y3 = self.lmList[p3] 
        
        # Calculate angle 
        angle = math.degrees(math.atan2(y3-y2,x3-x2) -
                        math.atan2(y1-y2,x1-x2))
        if angle < 0:
            angle += 360
        
        # Correct the angle to always represent the internal angle at the joint
        # Assuming we want the angle to be between 0 and 180 degrees
        if angle > 180:
            angle = 360 - angle
        
        # Draw
        if draw:
            cv2.line(img, (x1, y1),(x2, y2),(255,255,255), 3)
            cv2.line(img, (x3, y3),(x2, y2),(255,255,255), 3)
            cv2.circle(img,(x1,y1), 10, (0,0,255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0,0,255), 2)
            cv2.circle(img,(x2,y2), 10, (0,0,255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0,0,255), 2)
            cv2.circle(img,(x3,y3), 10, (0,0,255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0,0,255), 2)
        
            cv2.putText(img, str(int(angle)), (x2-20,y2+50),
                        cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)
        return angle
    '''
    


#STEP 2: testing script
def main():
    cap = cv2.VideoCapture('../vid/squat3.mp4')
    detector = poseDetector()
    '''
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        print(lmList)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
    '''
    

#STEP 1: this function will say : if this file alone is being run, it will run but
# when a function in this file is being called, it will only run that specific called function
if __name__ == "__main__":
    main()