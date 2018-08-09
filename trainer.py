import os
import cv2
import numpy as np
from PIL import Image

recognizer1 = cv2.face.FaceRecognizer()
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
path = '/Users/kushalgupta/face_detection_opencv/dataSet'

def getImageWithId(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    Ids=[]
    for imagePath in imagePaths:
        pilImage=Image.open(imagePath).convert('L')
        imageNp=np.array(pilImage,'uint8')
        Id=os.path.split(imagePath)[-1].split(".")[1]
 #       faceSamples.append(imageNp)
#        Ids.append(Id)
        faces=detector.detectMultiScale(imageNp)
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
  
 #       cv2.waitKey(10)
    return Ids,faceSamples

ids,faces = getImageWithId(path)
detector.train(faces,np.array(ids)
#recognizer1.save('recognizer/trainner.yml')
recognizer1.write('/Users/kushalgupta/face_detection_opencv/recognizer/trainingData.yml')
cv2.destroyAllWindows()
               

 

