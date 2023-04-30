import argparse
import json
import os
import time
import importlib.util
from threading import Thread

import cv2
import numpy as np

from twilio.rest import Client
import twilio
import geocoder


MODEL_NAME = 'custom_model_lite'
GRAPH_NAME = 'detect.tflite'
LABEL_MAP_NAME = 'labelmap.txt'
MIN_CONF_THRESHOLD = float(0.70)
USE_TPU = False
RESOLUTION = (1280, 720)
FRAMERATE = 30
im_W, im_H = int(1280), int(720)


CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABEL_MAP_NAME)

with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

if labels[0] == '???':
    del(labels[0])

def send_mms():
    now = time.strftime("%H:%M:%S", time.localtime())
    time_now = str(now)
    account_sid = "AC3091ae3709ee033a245b27f8596292bf"
    auth_token = "6044ef840e5bb51f06109a3d387a862a"
    client = Client(account_sid, auth_token)

    message = client.messages \
        .create(
            body='ALERT: GUN DETECTED \n Camera #: 1 \n Time: ' + time_now,
            media_url='https://t3.ftcdn.net/jpg/03/21/62/56/360_F_321625657_rauGwvaYjtbETuwxn9kpBWKDYrVUMdB4.jpg',
            from_='+18559475695',
            to='+16237766223'
        )

    print(message.sid)

class VideoStream:
    def __init__(self, resolution=RESOLUTION, framerate=FRAMERATE):
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])
            
        (self.grabbed, self.frame) = self.stream.read()

        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return

            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


if USE_TPU:
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if USE_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if USE_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

if USE_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    outname = output_details[0]['name']

    if ('StatefulPartitionedCall' in outname):
        boxes_idx, classes_idx, scores_idx = 1, 3, 0
    else:
        boxes_idx, classes_idx, scores_idx = 0, 
    
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    videostream = VideoStream(resolution=RESOLUTION, framerate=FRAMERATE).start()
    time.sleep(1)

    # Load config file
    with open("config.json", 'r') as file:
        config = json.load(file)
    last_detection_time = None   

    while True:
        t1 = cv2.getTickCount()
        frame1 = videostream.read()
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))

        input_data = np.expand_dims(frame_resized, axis=0)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

        for i in range(len(scores)):
            current_time = time.strftime("%H:%M:%S", time.localtime())
            if ((scores[i] > MIN_CONF_THRESHOLD) and (scores[i] <= 70.0)):
                y_min = int(max(1,(boxes[i][0] * im_H)))
                x_min = int(max(1,(boxes[i][1] * im_W)))
                y_max = int(min(im_H,(boxes[i][2] * im_H)))
                x_max = int(min(im_W,(boxes[i][3] * im_W)))
                cv2.rectangle(frame, (x_min,y_min), (x_max,y_max), (10, 255, 0), 2)

                object_names = labels[int(classes[i])]
                label = f'{object_names}: {int(scores[i]*100)}%'
                label_size, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(y_min, label_size[1] + 10)
                cv2.rectangle(frame, (x_min, label_ymin-label_size[1]-10), (x_min+label_size[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (x_min, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                if last_detection_time is None or time.time() - last_detection_time >= 30:
                    print("Object detected")
                    send_mms()
                    last_detection_time = time.time()
                    config['time'] = current_time
                    with open('config.json', 'w') as file:
                        json.dump(config, file, indent=4)
                    
        cv2.putText(frame, f'FPS: {frame_rate_calc:.2f}', (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)

        cv2.imshow('Object Detector', frame)

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1
        current_object_count = 1

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    videostream.stop()

