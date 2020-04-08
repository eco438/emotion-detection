import face_recognition
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

classifier = load_model('Emotion_little_vgg.h5')
class_labels = ['angry','happy','neutral','sad','surprise']


while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        
    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left) in face_locations :
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 5)
        roi = frame[top:bottom,left:right]
        roi = cv2.resize(roi,(48,48),interpolation=cv2.INTER_AREA)

        if np.sum([frame])!= 0:
            ro = roi.astype('float')/255.0
            ro = img_to_array(ro)
            ro = np.expand_dims(ro,axis=0)

        # Draw a label with a name below the face
            preds =classifier.predict(ro)[0]
            label = class_labels[preds.argmax()]
            label_position = (top,right)
            cv2.putText(frame,label_position,cv2.FRONT_HERSHEY_SIMPLEX,2,(255,0,0),3)

        
        

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
