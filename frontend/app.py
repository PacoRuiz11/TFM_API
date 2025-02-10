import asyncio
import base64
import threading
import streamlit as st
import requests
import time 
import io
from PIL import Image
import cv2
import numpy as np
import websockets



st.set_page_config(page_title="Object Detection App", 
                   layout="wide",
                   initial_sidebar_state="expanded",
                   page_icon="📸")

st.markdown(
    """
    <style>
    :root {
        --primary-color: #1e1e1e;
        --background-color: #0e1117;
        --secondary-background-color: #262730;
        --text-color: #fafafa;
        --font: 'Source Sans Pro', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)





API_BASE_URL = "http://localhost:8080/prediction"

def upload_image(image_file, model_type):
    files = {'file' : image_file}
    model = {"model_type": model_type}
    response = requests.post(f'{API_BASE_URL}/uploadimage', files=files, params=model)
    return response.json()["image_id"]

def upload_video(video_file, model_type):
    files = {'file': video_file}
    model = {"model_type": model_type}
    response = requests.post(f'{API_BASE_URL}/uploadvideo', files=files, params=model) 
    return response.json()["video_id"]

def get_status_image(image_id):
    response = requests.get(f'{API_BASE_URL}/status_image/{image_id}')
    return response.json()["status"]

def get_status_video(video_id):
    response = requests.get(f'{API_BASE_URL}/status_video/{video_id}')
    return response.json()["status"]

def get_result_image(image_id):
    response = requests.get(f'{API_BASE_URL}/result_image/{image_id}', stream=True)
    if response.status_code == 200:
        return response.content
    else:
        return None
    
def get_result_video(video_id):
    response = requests.get(f'{API_BASE_URL}/result_video/{video_id}', stream=True)
    if response.status_code == 200:
        return response.content
    else:
        return None
    
def delete_image(image_id):
    response = requests.delete(f'{API_BASE_URL}/image/{image_id}')
    return response.json()

def delete_video(video_id):
    response = requests.delete(f'{API_BASE_URL}/video/{video_id}')
    return response.json()


async def stream_webcam():
    ws_url = f"ws://localhost:8080/prediction/ws/predict_realtime?model_type={st.session_state.get('model', 'sesame')}"
    
    async with websockets.connect(ws_url) as ws:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Error: Could not open webcam.")
            return

        video_placeholder = st.empty()  # Placeholder for streaming frames

        while st.session_state.get("realtime", False):
            ret, frame = cap.read()
            if not ret:
                break
            
            _, buffer = cv2.imencode(".jpg", frame)
            encoded_frame = base64.b64encode(buffer).decode("utf-8")
            await ws.send(encoded_frame)

            response = await ws.recv()
            frame = base64.b64decode(response)
            nparr = np.frombuffer(frame, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Stream frame dynamically
            video_placeholder.image(frame, channels="BGR", use_container_width=True)

        cap.release()







# Sidebar
st.sidebar.title("Control Panel")
st.sidebar.subheader("Upload your file here")
upload_type = st.sidebar.radio("Choose upload type", ["Image", "Video", "Real-time Video"])
model_type = st.sidebar.radio("Choose model" , ["sesame", "pepper"])

if upload_type == "Image":
    image_file = st.sidebar.file_uploader("Upload file", type=["jpg", "jpeg", "png"])
    if image_file:
        st.sidebar.write("Press the button to upload the image:")
        if st.sidebar.button("Upload image"):
            try:
                image_id = upload_image(image_file, model_type)
                st.session_state["id"] = image_id
                st.session_state["is_image"] = True
                st.session_state["model"] = model_type
                st.sidebar.success("Image uploaded successfully")
            except Exception as e:
                st.error(f"An error occurred: {e}")

elif upload_type == "Video":
    video_file = st.sidebar.file_uploader("Upload file", type=["mp4", "avi", "mov"])
    if video_file:
        st.sidebar.write("Press the button to upload the video:")
        if st.sidebar.button("Upload video"):
            try:
                video_id = upload_video(video_file, model_type)
                st.session_state["id"] = video_id
                st.session_state["is_image"] = False
                st.session_state["model"] = model_type
                st.sidebar.success("Video uploaded successfully")
            except Exception as e:
                st.error(f"An error occurred: {e}")

elif upload_type == "Real-time Video":
    if st.sidebar.button("Start Real-time Video"):
        st.session_state["realtime"] = True
                


# Main page
st.title("Crop detection App")

if "id"  in st.session_state:
    pred_id = st.session_state["id"]
    is_image = st.session_state.get("is_image", True)

    get_status_func = get_status_image if is_image else get_status_video
    status_container = st.empty()
    result_container = st.empty()

    with st.spinner("Waiting for the prediction to finish..."):
        status = ""
        for _ in range(100):
            try:
                status = get_status_func(pred_id)
                if status == "Completed":
                    status_container.success("Prediction completed")
                    break
                elif status == "Failed":
                    status_container.error("Prediction failed")
                    break
                else:
                    status_container.info(f"Prediction status: {status}")
                    time.sleep(2)
            except Exception as e:
                status_container.error(f"An error occurred: {e}")
                break

    if status == "Completed":
        if is_image:
            st.write("Result of the image prediction:")
            try:
                result_image = get_result_image(pred_id)
                result_image = Image.open(io.BytesIO(result_image))
                resize_image = result_image.resize((500, 500))

                with result_container:
                    st.image(resize_image, caption="Image prediction", width=500)

                img_bytes = io.BytesIO()
                result_image.save(img_bytes, format="JPEG")
                img_bytes.seek(0)

                st.download_button(
                    label="Download Image Prediction",  
                    data=img_bytes,
                    file_name="image_result.jpg",  
                    mime="image/jpeg"
                )
            except Exception as e:
                st.error(f"An error occurred: {e}")
        
        else:
            st.write("Result of the video prediction:")
            try:
                result_video = get_result_video(pred_id)
                
                video_bytes = io.BytesIO(result_video)
                video_bytes.seek(0)  

                with result_container:
                    st.video(result_video, format="video/mp4")


                st.download_button(
                    label="Download Video Prediction",  
                    data=video_bytes,
                    file_name="video_result.mp4",  
                    mime="video/mp4"
                )

            except Exception as e:
                st.error(f"An error occurred: {e}")


# Handle Real-time Video Processing
if "realtime" in st.session_state and st.session_state["realtime"]:
    st.write("Real-time Object Detection Running...")

    # Run the async function in a proper context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(stream_webcam())
    loop.close()

