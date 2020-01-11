import cv2
import numpy as np
from base_camera import BaseCamera


class Camera(BaseCamera):
    video_source = 0

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, img = camera.read()

            imgShape = img.shape

            color =  (0,255,0)
            thickness =  3
            center = np.array([imgShape[1]/2, imgShape[0]/2])
            d =  np.array([128,128])
            p1 = tuple((center-d).astype(int))
            p2 = tuple((center+d).astype(int))
            cv2.rectangle(img,p1,p2,color,thickness)

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', img)[1].tobytes()
