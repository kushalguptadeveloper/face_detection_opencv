import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.createLBPHFaceRecognizer()
path = '/Users/kushalgupta/face_detection_opencv/dataSet'

def getImageWithId(path):
    
