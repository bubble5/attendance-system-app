import streamlit as st

st.set_page_config(page_title='Attendance System')
st.header('Attendance System using Face Recognition')

with st.spinner("Loadind Model and connecting to Redis DB"):
    import face_rec
st.success("Model Loaded successfully")
st.success("Redis DB successfully connected")