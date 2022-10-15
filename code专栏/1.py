
from array import array
from re import I


while 0:
    #coding=utf-8
    import cv2
    cap=cv2.VideoCapture(0) #cv2.VideoCapture(0)代表调取摄像头资源，其中0代表电脑摄像头，1代表外接摄像头(usb摄像头)
    cap.set(3,640)#宽
    cap.set(4,480)#高
    cap.set(10,100)#亮度
    while True:
        success,img=cap.read()
        # print(img,success)
        cv2.imshow("Video",img)
        if cv2.waitKey(1)&0xFF==ord('q'):
            break

while 1:#交互式标注
    import pylab
    from PIL import Image
    import numpy as np
    img = np.array(Image.open('/home/ktd/rpf_ws/cv_learn/img/auto.png'))
    pylab.imshow(img)
    print('Please click 3 points')
    x = pylab.ginput(3)
    print('you clicked :',x)
    pylab.show()