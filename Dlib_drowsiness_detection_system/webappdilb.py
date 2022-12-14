
import cv2
import numpy as np
import dlib
from imutils import face_utils
import streamlit as st
import base64
from pygame import mixer

import sys

def runk():
    mixer.init()
    sound = mixer.Sound('alarm.wav')

    cap = cv2.VideoCapture(0)

#Initializing the face detector and landmark detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks_GTX.dat")

#status marking for current state
    sleep = 0
    drowsy = 0
    active = 0
    status=""
    color=(0,0,0)
    FRAME_WINDOW = st.image([])

    def compute(ptA,ptB):
        dist = np.linalg.norm(ptA - ptB)
        return dist

    def blinked(a,b,c,d,e,f):
        up = compute(b,d) + compute(c,e)
        down = compute(a,f)
        y7u7yratio = up/(2.0*down)

    #Checking if it is blinked
        if(ratio>0.25):
            return 2
        elif(ratio>0.21 and ratio<=0.25):
            return 1
        else:
            return 0

    while True:
        ret, frame = cap.read()
        if not ret:
           print("Can't receive frame (stream end?). Exiting ...")
           break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            face_frame = frame.copy()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (1, 1, 1), 2)

            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            left_blink = blinked(landmarks[36],landmarks[37], 
            landmarks[38], landmarks[41], landmarks[40], landmarks[39])
            right_blink = blinked(landmarks[42],landmarks[43], 
            landmarks[44], landmarks[47], landmarks[46], landmarks[45])
        
        #Now judge what to do for the eye blinks
            if(left_blink==0 or right_blink==0):
                sleep+=1
                drowsy=0
                active=0
                if(sleep>6):
                    status="SLEEPING !!!"
                    color = (255,0,0)

            elif(left_blink==1 or right_blink==1):
                sleep=0
                active=0
                drowsy+=1
                if(drowsy>6):
                    status="Drowsy !"
                    color = (0,0,255)

            else:
                drowsy=0
                sleep=0
                active+=1
                if(active>6):
                    status="Active :)"
                    color = (0,255,0)
        
            cv2.putText(frame, status, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color,3)
            # putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]]) -> img
            for n in range(0, 68):
                (x,y) = landmarks[n]
                cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)
    
        cv2.imshow("Frame", frame)
#     cv2.imshow("Result of detector", face_frame)
        key = cv2.waitKey(10)
        if key == ord('q'):
            break       
    # frame.release()
    cap.release()
    cv2.destroyAllWindows()

def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
        )

def main():
    """Drowsiness Detection App"""
    add_bg_from_local("pic-2.png")
    st.title("Drowsiness Detection App")
    st.subheader('Welcome to the Drowsiness Detection App')
    if st.button('Please start the detection'):
        st.write('Drowsiness Detection Started')
        runk()
        
        #st.text("Drowsiness Detection Started")
    

if __name__ == '__main__':
    main()