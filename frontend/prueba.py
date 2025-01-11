import streamlit as st

video_file = open("C:/Users/Paco/Desktop/MASTER BIG DATA/5. TFM/proyecto/model/Inference/output/predictions/ajonjoli_predictions_fixed.mp4", "rb")
video_bytes = video_file.read()

st.video(video_bytes)