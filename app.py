#!/usr/bin/env python
import os
import cv2
import numpy as np
from importlib import import_module
from flask import Flask, render_template, Response
from camera_opencv import Camera



app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen(camera):
    """Video streaming generator function."""
    while True:
        img = camera.get_frame()
        #img = cv2.flip(img, +1)

        imgShape = img.shape

        '''color =  (0,255,0)
        thickness =  3
        center = np.array([imgShape[1]/2, imgShape[0]/2])
        d =  np.array([128,128])
        p1 = tuple((center-d).astype(int))
        p2 = tuple((center+d).astype(int))
        cv2.rectangle(img,p1,p2,color,thickness)'''

        # encode as a jpeg image and return it
        frame = cv2.imencode('.jpg', img)[1].tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
