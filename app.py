#!/usr/bin/env python
import os
import cv2
import numpy as np
from flask import Flask, render_template, Response
from camera_opencv import Camera
import argparse
import tensorflow as tf
from datetime import datetime

parser = argparse.ArgumentParser()

parser.add_argument('--debug', action='store_true',help='Wait for debugge attach')
parser.add_argument('--model', type=str, default='./saved_model/1590670812',
                    help='Base directory for the model.')


_HEIGHT = 200
_WIDTH = 200
_DEPTH = 3

app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen(camera):
    """Video streaming generator function."""

    #loaded = tf.saved_model.load_v2(FLAGS.model)
    loaded = tf.saved_model.load(FLAGS.model)
    print(list(loaded.signatures.keys()))
    infer = loaded.signatures["serving_default"]
    print(infer.structured_outputs)
    print (infer.inputs[0])

    while True:
        img = camera.get_frame()
        #img = cv2.flip(img, +1)

        imgShape = img.shape
        #Define crop of 2x CNN size and downsize it in tf.image.crop_and_resize
        color =  (0,255,0)
        thickness =  3
        center = np.array([imgShape[1]/2, imgShape[0]/2])
        d =  np.array([_HEIGHT/2,_WIDTH/2])
        p1 = tuple((center-d).astype(int))
        p1 = (max(p1[0],0),max(p1[1],0))
        p2 = tuple((center+d).astype(int))
        p2 = (min(p2[0],imgShape[0]-1),min(p2[1],imgShape[1]-1))
        cv2.rectangle(img,p1,p2,color,thickness)
        crop = cv2.resize(img[p1[1]:p2[1], p1[0]:p2[0]],(_WIDTH,_HEIGHT)).astype(np.float32)

        before = datetime.now()

        outputs = infer(tf.constant(crop))
        gender = 'male'
        pred_gender = np.squeeze(outputs['pred_gender'].numpy())
        pred_age = np.squeeze(outputs['pred_age'].numpy())
        dt = datetime.now()-before

        outputs['pred_age'].numpy()
        if(outputs['pred_gender'].numpy()[0] < 1):
            gender = 'female'
        results = 'Age {}, Genderender {}, '.format(outputs['pred_age'].numpy()[0,0],outputs['pred_gender'].numpy()[0])


        resultsDisplay = '{:.3f}s Age {}, Gender {}, '.format(dt.total_seconds(), int(round(outputs['pred_age'].numpy()[0,0])),gender)

        print(results)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, resultsDisplay, (10,25), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
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
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.debug:
        # https://code.visualstudio.com/docs/python/debugging#_remote-debugging
        # Launch applicaiton on remote computer: 
        # > python3 -m ptvsd --host 0.0.0.0 --port 3000 --wait predict_imdb.py
        import ptvsd
        # Allow other computers to attach to ptvsd at this IP address and port.
        ptvsd.enable_attach(address=('0.0.0.0', 3000), redirect_output=True)
        # Pause the program until a remote debugger is attached
        print("Wait for debugger attach")
        ptvsd.wait_for_attach()
        print("Debugger Attached")

    print(tf.version)

    app.run(host='0.0.0.0', threaded=True)
