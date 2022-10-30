from asyncore import ExitNow
import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
import streamlit as st
import base64


#st.title("Drowsiness Detection App")
def runk():
    mixer.init()
    sound = mixer.Sound('alarm.wav')

    face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
    leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
    reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')



    lbl=['Close','Open']

    model = load_model('models/cnnCat2.h5')
    path = os.getcwd()
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    count=0
    score=0
    thicc=2
    rpred=[99] # list with n=integer element
    lpred=[99]
    FRAME_WINDOW = st.image([])
    #st.write('Drowsiness Detection Started')
    while(True):
        # FRAME_WINDOW = st.image([])
        ret, frame = cap.read()
        height,width = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
        left_eye = leye.detectMultiScale(gray)
        right_eye =  reye.detectMultiScale(gray)

        cv2.rectangle(frame, (0,height-30) , (540,height) , (0,0,0) , thickness=cv2.FILLED ) # rectangle getting below

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 ) # face rectangle

        for (x,y,w,h) in right_eye:
            r_eye=frame[y:y+h,x:x+w]
            count=count+1
            r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye,(24,24))
            r_eye= r_eye/255
            r_eye=  r_eye.reshape(24,24,-1)
            r_eye = np.expand_dims(r_eye,axis=0)
            rpred = model.predict(r_eye)
            if(np.argmax(rpred[0])==1):
                lbl='Open' 
            if(np.argmax(rpred[0])==0):
                lbl='Closed'
            break

        for (x,y,w,h) in left_eye:
            l_eye=frame[y:y+h,x:x+w]
            count=count+1
            l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
            l_eye = cv2.resize(l_eye,(24,24))
            l_eye= l_eye/255
            l_eye=l_eye.reshape(24,24,-1)
            l_eye = np.expand_dims(l_eye,axis=0)
            lpred = model.predict(l_eye)
            if(np.argmax(lpred[0])==1):
                lbl='Open'   
            if(np.argmax(lpred[0])==0): 
                lbl='Closed'
            break

        if(np.argmax(rpred[0])==0 and np.argmax(lpred[0])==0):
            score=score+1
            cv2.putText(frame,"Closed!!! Open your eyes",(30,height-10), font, 1,(255,255,255),2,cv2.LINE_AA) # making font thiker/bold
        # if(rpred[0]==1 or lpred[0]==1):
        else:
            score=score-1
            cv2.putText(frame,"Open:-) keep going",(30,height-10), font, 1,(255,255,255),2,cv2.LINE_AA)
        
            
        if(score<0):
            score=0   
        cv2.putText(frame,'Score:'+str(score),(400,height-10), font, 1,(255,255,255),2,cv2.LINE_AA)
        if(score>15):
            #person is feeling sleepy so we beep the alarm
            cv2.imwrite(os.path.join(path,'image.jpg'),frame)
            try:
                sound.play()
                
            except:  # isplaying = False
                pass
            if(thicc<16):
                thicc= thicc+2
            else:
                thicc=thicc-2
                if(thicc<2):
                    thicc=2
            cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) # window frame if we inc this then window will show border
        #cv2.imshow('frame',frame)
        FRAME_WINDOW.image(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    #if st.button('close web cam'):
     #   ExitNow

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

