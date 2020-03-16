# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

# ASL Detector code
import numpy as np
import cv2
import torch
from model import Net

modelo = torch.load('trained_model.pt')

dict_labels = {'0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G', '7': 'H', '8': 'I',
        '10': 'K', '11': 'L', '12': 'M', '13': 'N', '14': 'O', '15': 'P', '16': 'Q', '17': 'R',
        '18': 'S', '19': 'T', '20': 'U', '21': 'V', '22': 'W', '23': 'X', '24': 'Y' }

# import the necessary packages
from imutils.video import VideoStream
from flask import Response, request
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2
from flask import jsonify
import autocomplete

autocomplete.load()

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()
trigger_flag = False
total_str = ''

# initialize a flask object
app = Flask(__name__)

# initialize the video stream and allow the camera sensor to
# warmup
vc = VideoStream(src=0).start()
time.sleep(2.0)

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")
            
def detect_gesture(frameCount):

    global vc, outputFrame, lock, trigger_flag, total_str

    # vc = cv2.VideoCapture(0)
    # rval, frame = vc.read()
    old_text = ''
    pred_text = ''
    flag = False

    while True:
        frame = vc.read()
        
        if frame is not None: 

            frame = cv2.flip(frame, 1)
            width = 720
            height = 480
            
            frame = cv2.resize( frame, (width,height))

            cv2.rectangle(frame, (width//2, 0), (width, width//2), (0,255,0), 2)

            crop_img = frame[0:width//2, width//2:width]
            grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

            thresh = cv2.threshold(grey,210,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

            # blackboard = np.zeros(frame.shape, dtype=np.uint8)
            cv2.putText(frame, "Predicted letter - ", (30, 40), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))

            if trigger_flag == True:
                old_text = pred_text
                thresh_resized = cv2.resize(thresh, (50,50))
                pred_text = predictor(thresh_resized)

                if(pred_text=='nothing' or pred_text=='space'):
                    total_str += ' '
                else:
                    total_str += pred_text

                trigger_flag = False
            
            cv2.putText(frame, pred_text, (30, 80), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 127))
            
            # res = np.hstack((frame, blackboard))
            with lock:
                outputFrame = frame.copy()

		
def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock

	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue

		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')


def get_suggestion(prev_word='my', next_semi_word='na'):
    global total_str

    separated = list(total_str)

    first = ''
    second = ''

    if(len(separated)>=2):
        first = separated[-2]
        second = separated[-1]

    if(first and second):
        suggestions = autocomplete.predict(first, second)[:5]
    else:
        suggestions = autocomplete.predict(total_str, '')[:5]
    return [word[0] for word in suggestions]

@app.route('/trigger') 
def form_example():
    global trigger_flag
    trigger_flag = True
    return Response('done')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route('/suggestions')
def suggestion():
    suggestions = get_suggestion()
    return jsonify(suggestions)

@app.route('/sentence')
def sentence():
    global total_str
    return jsonify(total_str)

# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	args = vars(ap.parse_args())

	# start a thread that will perform motion detection
	t = threading.Thread(target=detect_gesture, args=(
		args["frame_count"],))
	t.daemon = True
	t.start()

	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)

# release the video stream pointer
# vs.stop()