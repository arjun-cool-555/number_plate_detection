import cv2
import numpy as np
import easyocr
from PIL import Image
from flask import Flask,request,app,render_template

app=Flask(__name__)

def text_detection(image):
  n_plate_detector = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
  img = np.asarray(image)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  detections = n_plate_detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7)
  if len(detections)==1:
    for (x,y,w,h) in detections:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    plate = img[y:y + h, x:x + w]

    reader = easyocr.Reader(['en'])
    result = reader.readtext(plate)
    return "The predicted Text is "+result[0][1]
  else:
    return "Sorry, model unable to detect Text"

@app.route('/',methods=["GET"])
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    file = request.files["file"]
    img = Image.open(file.stream)
    output=text_detection(img)
    return render_template('index.html',prediction_text=output)

if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0",port="5000")