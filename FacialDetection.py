import face_recognition
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
# Get a reference to webcam #0 (the default one)



video_capture = cv2.VideoCapture(0)

# Initialize some variables
face_locations = []
face_encodings = []


classifier =load_model('Emotion_model.h5')
emotion_dict = {0: "Angry",  2: "Happy", 3: "Neutral", 4: "Sad", 5: "Surprised"}

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    (H, W) = frame.shape[:2]
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255,0), 2)
        roi_gray = gray[top:bottom,left:right]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        # Draw a label with a name below the face
        if np.sum([roi_gray])!=0:
        # make a prediction on the ROI, then lookup the class
            cropped_img = np.expand_dims(np.expand_dims(roi_gray,-1),0)
            preds = classifier.predict(cropped_img)
            label="Emotion: "+emotion_dict[int(np.argmax(preds))]  
            label_position = (left,bottom+25)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(10,H),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)
    cv2.imshow('Emotion Detector',frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
