
import streamlit as st
from Home import face_rec

#st.set_page_config(page_title='Report')
st.subheader('Reports')

#Retrieve the logs from Redis DB
name= "attendance:logs"
def load_logs(name):
    logs=face_rec.r.lrange(name,0,-1)
    return logs

tab1,tab2=st.tabs(["Registered Users", "User Logs"])
with tab1:
    with st.spinner("Retrieving Data from Redis DB...."):
        redis_face_db=face_rec.retrieve_data(name='academy:register')
        st.dataframe(redis_face_db[['name','role']])

with tab2:
    if st.button("Refresh Logs"):
        st.write(load_logs(name=name))