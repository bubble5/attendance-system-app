
import streamlit as st
from Home import face_rec
from streamlit_webrtc import webrtc_streamer
import av
import time

#st.set_page_config(page_title='Real Time Prediction')
st.subheader('Real Time Attendance System')

#Retieve the data from redis Database

with st.spinner("Retrieving Data from Redis DB...."):
    redis_face_db=face_rec.retrieve_data(name='academy:register')
    st.dataframe(redis_face_db)
st.success("Data successfully retrieved from Redis")
#time
waittime = 30 #time in secs
setTime = time.time()
realtimepred = face_rec.RealTimePred() # real time prediction class


#Real Time Prediction using stremlit webrtc



def video_frame_callback(frame):
    global setTime

    img = frame.to_ndarray(format="bgr24") #3Dimension Array
    # Operation that you can perform on the array
    pred_img=realtimepred.face_prediction(img,redis_face_db,'facial_features',
                                      ['name', 'role'],thresh=0.5)
    
    timenow = time.time()
    difftime = timenow - setTime

    if difftime >= waittime :
        realtimepred.saveLogs_redis()
        setTime = time.time() #reset time
        

        print('Save data to redis DB')
        st.success("Data saved successfully")


    

    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")


webrtc_streamer(key="realtimePrediction", video_frame_callback=video_frame_callback,
                  rtc_configuration={
                         "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                                     } 
                )
