# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

# import the necessary packages
from pyimagesearch.motion_detection import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
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
import numpy as np

autocomplete.load()

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

# initialize the video stream and allow the camera sensor to
# warmup
# vs = VideoStream(src=0).start()
# time.sleep(2.0)

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

def detect_motion(frameCount):
	# grab global references to the video stream, output frame, and
	# lock variables
	global vs, outputFrame, lock

	# initialize the motion detector and the total number of frames
	# read thus far
	md = SingleMotionDetector(accumWeight=0.1)
	total = 0

	# loop over frames from the video stream
	while True:
		# read the next frame from the video stream, resize it,
		# convert the frame to grayscale, and blur it
		frame = vs.read()
		frame = imutils.resize(frame, width=400)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (7, 7), 0)

		# grab the current timestamp and draw it on the frame
		timestamp = datetime.datetime.now()
		cv2.putText(frame, timestamp.strftime(
			"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

		# if the total number of frames has reached a sufficient
		# number to construct a reasonable background model, then
		# continue to process the frame
		if total > frameCount:
			# detect motion in the image
			motion = md.detect(gray)

			# cehck to see if motion was found in the frame
			if motion is not None:
				# unpack the tuple and draw the box surrounding the
				# "motion area" on the output frame
				(thresh, (minX, minY, maxX, maxY)) = motion
				cv2.rectangle(frame, (minX, minY), (maxX, maxY),
					(0, 0, 255), 2)
		
		# update the background model and increment the total number
		# of frames read thus far
		md.update(gray)
		total += 1

		# acquire the lock, set the output frame, and release the
		# lock
		with lock:
			outputFrame = frame.copy()
            
def detect_gesture(frameCount):
    vc = cv2.VideoCapture(0)
    rval, frame = vc.read()
    old_text = ''
    pred_text = ''
    count_frames = 0
    total_str = ''
    flag = False

    while True:
    
        if frame is not None: 

            frame = cv2.flip(frame, 1)
            frame = cv2.resize( frame, (400,400) )

            cv2.rectangle(frame, (300,300), (100,100), (0,255,0), 2)

            crop_img = frame[100:300, 100:300]
            grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

            thresh = cv2.threshold(grey,210,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]


            blackboard = np.zeros(frame.shape, dtype=np.uint8)
            cv2.putText(blackboard, "Predicted text - ", (30, 40), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
#             if count_frames > 20 and pred_text != "":
#                 total_str += pred_text
#                 count_frames = 0

#             if flag == True:
#                 old_text = pred_text
#                 pred_text = predictor(thresh)

#                 if old_text == pred_text:
#                     count_frames += 1
#                 else:
#                     count_frames = 0
#                 cv2.putText(blackboard, total_str, (30, 80), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 127))
            res = np.hstack((frame, blackboard))

            cv2.imshow("image", res)
            cv2.imshow("hand", thresh)

        rval, frame = vc.read()
        keypress = cv2.waitKey(1)
        if keypress == ord('c'):
            flag = True
        if keypress == ord('q'):
            break

    vc.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    vc.release()
 
    
    
		
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
    suggestions = autocomplete.predict(prev_word, next_semi_word)[:5]
    return [word[0] for word in suggestions]

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
    sentence = 'I want to buy some apples...'
    return jsonify(sentence)

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