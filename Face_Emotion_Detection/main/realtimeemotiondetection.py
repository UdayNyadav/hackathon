import cv2  
from keras.models import model_from_json
import numpy as np

json_file = open( r"C:\Users\udayn\Code\Face_Emotion_Detection\emotiondetector1.json", 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights(r"C:\Users\udayn\Code\Face_Emotion_Detection\emotiondetector1.h5")
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)
emotion_count = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'neutral': 0, 'sad': 0, 'surprise': 0}

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

def detect_emotion():
    webcam = cv2.VideoCapture(0)  # Use DirectShow instead of MSMF
    labels = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4:'neutral', 5:'sad', 6:'surprise'}
    prediction_label = None
    count = 0
    while True:
        i, im= webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # faces = face_cascade.detectMultiScale(im, 1.3, 5)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Use gray image

        try:
            for (p,q,r,s) in faces:
                image = gray[q:q+s, p:p+r]
                cv2.rectangle(im, (p,q), (p+r, q+s), (255, 0, 0), 2)
                image = cv2.resize(image, (48,48))
                img = extract_features(image)
                pred = model.predict(img)
                prediction_label = labels[pred.argmax()]
                cv2.putText(im, '%s' %(prediction_label), (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,255,0))
            cv2.waitKey(27)
            cv2.imshow('output', im)   
        except cv2.error:
            pass
        

        if prediction_label != None:
            print('Emotion :'+prediction_label)
            emotion_count[prediction_label] = emotion_count[prediction_label]+1
            count = count +1
            if count == 150:
                break
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or count == 150:
            break

    webcam.release()
    cv2.destroyAllWindows()
    dominant_emotion = max(emotion_count, key= emotion_count.get)
    print(emotion_count)
    # print('most captured emotion:'+high)
    return dominant_emotion

# print(detect_emotion())