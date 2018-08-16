import cv2
import numpy as np
import sqlite3
faceDetect = cv2.CascadeClassifier('/anaconda3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

def getProfile(id):
    conn = sqlite3.connect("faceBase.db")
    cmd = "SELECT * FROM People Where Id="+str(id)
    cursor = conn.execute(cmd)
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile


cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read('/Users/kushalgupta/face_detection_opencv/recognizer/trainingData.yml')
id = 0
font = cv2.FONT_HERSHEY_SIMPLEX
#font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1) 
while(True):
    ret,img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        id,conf = rec.predict(gray[y:y+h,x:x+w])
        flip_fr = cv2.flip(img,1)
 ##       if id ==1:
##            id = "kushal"


        profile = getProfile(id)
        if profile!=None:
            cv2.putText(img,str(profile[1]),(x,y+h), font, 1, (200,255,155), 2, cv2.LINE_AA)
            

        
            
 ##       cv2.putText(img,str(id),(x,y+h), font, 1, (200,255,155), 2, cv2.LINE_AA)
#        cv2.putText(flip_fr, str(id), (x, y+h), font, 1,(255, 255, 0), 2)
 #       cv2.putText(cv2.cv.fromarray(img),str(id),(x,y+h),font,255,2)
    cv2.imshow("Face",img)
    if cv2.waitKey(1) == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
