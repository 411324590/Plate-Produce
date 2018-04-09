# -*- coding:utf-8 -*-
import os
import cv2
import numpy as np

for file in os.listdir("testImg"):
    filename = "testImg/"+file
    print(filename)
    img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    outfilename = "testImg02/"+file
    cv2.imencode('.jpg', img)[1].tofile(outfilename)
