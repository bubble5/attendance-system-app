
import streamlit as st
from Home import face_rec
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av


#st.set_page_config(page_title='Registration form')
st.subheader('Registration Form')

#Init registration form
registration_form=face_rec.RegistrationForm()

#Step1: Collect person name and role
person_name = st.text_input(label='Person Name',placeholder='first & last name')
role = st.selectbox(label='Person Role',options=('Teacher','Student'))

#Step2: Collect facial embeddings
def video_callback_func(frame):
    img = frame.to_ndarray(format='bgr24') #3d array bgr
    reg_img,embedding=registration_form.get_embedding(img)

    #To prepare data to save to redis we do that in two steps:
    #Step 1: save data into local computer in .txt
    if embedding is not None:
        with open('face_embedding.txt',mode='ab') as f:
            np.savetxt(f,embedding)

    

    return av.VideoFrame.from_ndarray(reg_img,format='bgr24')
webrtc_streamer(key='Registration',video_frame_callback=video_callback_func, 
                  rtc_configuration={
                     "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                     }
                
                )

#Step3: Save data to Redis DB
if st.button(label='Submit'):
    return_val=registration_form.save_data_in_redis_db(person_name,role)
    if return_val == True:
        st.success(f'{person_name} Registered successfully')
    elif return_val == 'Name is fale':
        st.error("Please enter the name: Name cannot be empty nor spaces")

    elif return_val == 'file_false':
        st.error("Embeddigs file is missig :file_embedding.txt Not found")