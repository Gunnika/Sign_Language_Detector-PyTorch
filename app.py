from imutils.video import VideoStream
from flask import Response, request
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
from flask import jsonify
import autocomplete

import cv2
import numpy as np
import torch
from model import Net

model = torch.load('model_trained.pt')
model.eval()

signs = {'0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G', '7': 'H', '8': 'I',
        '10': 'K', '11': 'L', '12': 'M', '13': 'N', '14': 'O', '15': 'P', '16': 'Q', '17': 'R',
        '18': 'S', '19': 'T', '20': 'U', '21': 'V', '22': 'W', '23': 'X', '24': 'Y' }

autocomplete.load()

outputFrame = None
lock = threading.Lock()
trigger_flag = False
full_sentence = ''
text_suggestion = ''

app = Flask(__name__)

vc = VideoStream(src=0).start()
time.sleep(2.0)
            
def detect_gesture(frameCount):

    global vc, outputFrame, lock, trigger_flag, full_sentence, text_suggestion

    while True:
        frame = vc.read()

        width = 700
        height = 480
            
        frame = cv2.resize( frame, (width,height))

        img = frame[20:250, 20:250]

        res = cv2.resize(img, dsize=(28, 28), interpolation = cv2.INTER_CUBIC)
        res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

        res1 = np.reshape(res, (1, 1, 28, 28)) / 255
        res1 = torch.from_numpy(res1)
        res1 = res1.type(torch.FloatTensor)

        out = model(res1)
        probs, label = torch.topk(out, 25)
        probs = torch.nn.functional.softmax(probs, 1)

        pred = out.max(1, keepdim=True)[1]

        if float(probs[0,0]) < 0.4:
            detected = 'Nothing detected'
        else:
            detected = signs[str(int(pred))] + ': ' + '{:.2f}'.format(float(probs[0,0])) + '%'

        if trigger_flag:
            full_sentence+=signs[str(int(pred))].lower()
            trigger_flag=False
        
        if(text_suggestion!=''):
            if(text_suggestion==' '):
                full_sentence+=' '
                text_suggestion=''
            else:
                full_sentence_list = full_sentence.strip().split()
                if(len(full_sentence_list)!=0):
                    full_sentence_list.pop()
                full_sentence_list.append(text_suggestion)
                full_sentence = ' '.join(full_sentence_list)
                full_sentence+=' '
                text_suggestion=''

        font = cv2.FONT_HERSHEY_SIMPLEX
        frame = cv2.putText(frame, detected, (60,285), font, 1, (255,0,0), 2, cv2.LINE_AA)

        frame = cv2.rectangle(frame, (20, 20), (250, 250), (0, 255, 0), 3)

        with lock:
            outputFrame = frame.copy()

		
def generate():
	global outputFrame, lock
	while True:
		with lock:
			if outputFrame is None:
				continue
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			if not flag:
				continue

		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')


def get_suggestion(prev_word='my', next_semi_word='na'):
    global full_sentence
    separated = full_sentence.strip().split(' ')

    print(separated)

    if(len(separated)==0):
        return ['i', 'me', 'the', 'my', 'there']
    elif(len(separated)==1):
        suggestions = autocomplete.predict(full_sentence, '')[:5]
    elif(len(separated)>=2):
        first = ''
        second = ''

        first = separated[-2]
        second = separated[-1]
        
        suggestions = autocomplete.predict(first, second)[:5]
        
    return [word[0] for word in suggestions]

@app.route("/")
def index():
	return render_template("index.html")

@app.route('/char') 
def char():
    global text_suggestion
    recommended = get_suggestion()
    option = request.args.get('character')
    if(option=='space'):
        text_suggestion=" "
    else:
        text_suggestion=recommended[int(option)-1]
    print(text_suggestion)
    return Response("done")

@app.route('/trigger') 
def trigger():
    global trigger_flag
    trigger_flag = True
    return Response('done')

@app.route("/video_feed")
def video_feed():
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route('/suggestions')
def suggestion():
    suggestions = get_suggestion()
    return jsonify(suggestions)

@app.route('/sentence')
def sentence():
    global full_sentence
    return jsonify(full_sentence)

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	args = vars(ap.parse_args())

	t = threading.Thread(target=detect_gesture, args=(
		args["frame_count"],))
	t.daemon = True
	t.start()

	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)

vc.stop()