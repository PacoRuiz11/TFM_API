from fastapi import APIRouter, HTTPException, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import os
from pydantic import BaseModel
from typing import Optional
import json
import uuid
from threading import Thread
import shutil as shuntil
import cv2
import tempfile
import subprocess


router = APIRouter(
    prefix="/prediction",
    tags=["prediction"],
    responses={404: {"description": "Not found"}},
)


MODEL_PATH_SESAME = 'C:/Users/Paco/Desktop/MASTER BIG DATA/5. TFM/proyecto/ML_API/backend/routers/best_sesame.pt'
MODEL_PATH_PEPPER = 'C:/Users/Paco/Desktop/MASTER BIG DATA/5. TFM/proyecto/ML_API/backend/routers/best_pepper.pt'

OUTPUT_PATH_IMAGES = 'outputs_images'
OUTPUT_PATH_VIDEOS = 'outputs_videos'


class ImageData(BaseModel):
    image_id: str
    image_path: Optional[str] = None
    status: str

    def save(self):
        if not os.path.exists(OUTPUT_PATH_IMAGES):
            os.makedirs(OUTPUT_PATH_IMAGES)
        file_path = os.path.join(OUTPUT_PATH_IMAGES, f"{self.image_id}.json")
        with open(file_path, "w") as f:
            f.write(self.model_dump_json(indent=4))
    
    @classmethod
    def load(cls, image_id: str):
        file_path = os.path.join(OUTPUT_PATH_IMAGES, F"{image_id}.json")
        with open(file_path, "r") as f:
            data = json.load(f)
            return cls(**data)


def load_model(model_type: str):
    if model_type == "sesame":
        model_path = MODEL_PATH_SESAME
    elif model_type == "pepper":
        model_path = MODEL_PATH_PEPPER
    else:
        raise ValueError("Invalid model type")
    model = YOLO(model_path)
    return model


def load_image(image_file):
    image = Image.open(io.BytesIO(image_file))
    return image



def predict_image(image, image_data: ImageData, model_type: str):
    try:
        model = load_model(model_type)
        results = model(image)
        os.makedirs(OUTPUT_PATH_IMAGES, exist_ok=True)
        image_path = os.path.join(OUTPUT_PATH_IMAGES, f"{image_data.image_id}.jpg")
        for result in results:
            result.save(image_path)
        image_data.image_path = image_path
        image_data.status = "Completed"
        image_data.save()
    except Exception as e:
        image_data.status = "Failed"
        image_data.save()
        raise HTTPException(status_code=400, detail=str(e))



@router.post("/uploadimage")
async def upload_image(file: UploadFile = File(...), model_type: str = "sesame"):
    try:
        image_bytes = await file.read()
        image = load_image(image_bytes)
        image_id = str(uuid.uuid4())
        image_data = ImageData(image_id=image_id, status="In Progress")
        image_data.save()
        print(model_type)
        Thread(target=predict_image, args=(image, image_data, model_type)).start()
        
        return {"image_id": image_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

@router.get("/status_image/{image_id}")
async def get_status(image_id: str):
    try:
        image_data = ImageData.load(image_id)

        if not image_data:
            raise HTTPException(status_code=404, detail="Image not found")

        return {"image_id": image_data.image_id, "status": image_data.status}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@router.get("/result_image/{image_id}")
async def get_result(image_id: str):
    try:
        image_data = ImageData.load(image_id)

        if image_data.status == "In Progress":
            return {"image_id": image_id, "status": "In Progress"}
        elif not image_data or not image_data.image_path:
            raise HTTPException(status_code=404, detail="Image not found")
        
        return FileResponse(image_data.image_path, media_type="application/octet-stream", filename=f"{image_id}.jpg")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/image/{image_id}")
async def delete_image(image_id: str):
    try:
        image_data = ImageData.load(image_id)

        if not image_data:
            raise HTTPException(status_code=404, detail="Image not found")
        
        if image_data.image_path and os.path.exists(image_data.image_path):
            os.remove(image_data.image_path)

        file_path = os.path.join(OUTPUT_PATH_IMAGES, f"{image_id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)

        return {"message": "Image deleted"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


#Process videos
class VideoData(BaseModel):
    video_id: str
    video_path: Optional[str] = None
    status: str

    def save_video(self):
        if not os.path.exists(OUTPUT_PATH_VIDEOS):
            os.makedirs(OUTPUT_PATH_VIDEOS)
        file_path = os.path.join(OUTPUT_PATH_VIDEOS, f"{self.video_id}.json")
        with open(file_path, "w") as f:
            f.write(self.model_dump_json(indent=4))
    
    @classmethod
    def load_video(cls, video_id: str):
        file_path = os.path.join(OUTPUT_PATH_VIDEOS, f"{video_id}.json")
        if not os.path.exists(file_path):
            return None
        with open(file_path, "r") as f:
            data = json.load(f)
            return cls(**data)


def predict_video(video_path, video_data: VideoData, model_type: str):
    try:
        model = load_model(model_type)

        if not os.path.exists(video_path):
            raise ValueError("Video file not found")
       

        results = model.predict(
            source=video_path,
            save=False,
            save_txt=False
        )

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        os.makedirs(OUTPUT_PATH_VIDEOS, exist_ok=True)
        video_path_original = os.path.join(OUTPUT_PATH_VIDEOS, f"{video_data.video_id}_original.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path_original, fourcc, fps, (width, height))

        for result in results:
            frame = result.plot()
            out.write(frame)
        
        out.release()

        # Re-codificación con FFmpeg
        fixed_output_path = os.path.join(OUTPUT_PATH_VIDEOS, f"{video_data.video_id}.mp4")
        ffmpeg_command = [
            "ffmpeg", "-i", video_path_original, "-vcodec", "libx264", "-acodec", "aac", fixed_output_path
        ]

        try:
            # Ejecutar el comando FFmpeg
            subprocess.run(ffmpeg_command, check=True)
            print(f"Video re-codificado guardado en: {fixed_output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error al ejecutar FFmpeg: {e}")

        video_data.video_path = fixed_output_path
        video_data.status = "Completed"
        video_data.save_video()        

    except Exception as e:
        video_data.status = "Failed"
        video_data.save_video()
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/uploadvideo")
async def upload_video(file: UploadFile = File(...), model_type: str = "sesame"):
    try:
        video_bytes = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(video_bytes)
            tmp_file.flush()
            video_path = tmp_file.name

        video_id = str(uuid.uuid4())
        video_data = VideoData(video_id=video_id, status="In Progress")
        video_data.save_video()
        Thread(target=predict_video, args=(video_path, video_data, model_type)).start()
       
        return {"video_id": video_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

@router.get("/status_video/{video_id}")
async def get_status_video(video_id: str):
    try:
        video_data = VideoData.load_video(video_id)

        if not video_data:
            raise HTTPException(status_code=404, detail="video not found")

        return {"video_id": video_data.video_id, "status": video_data.status}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

@router.get("/result_video/{video_id}")
async def get_result_video(video_id: str):
    try:
        video_data = VideoData.load_video(video_id)

        if video_data.status == "In Progress":
            return {"video_id": video_id, "status": "In Progress"}
        elif not video_data or not video_data.video_path:
            raise HTTPException(status_code=404, detail="video not found")
        
        return FileResponse(video_data.video_path, media_type="video/mp4", filename=f"{video_id}.mp4")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



@router.delete("/video/{video_id}")
async def delete_video(video_id: str):
    try:
        video_data = VideoData.load_video(video_id)

        if not video_data:
            raise HTTPException(status_code=404, detail="video not found")
        
        if video_data.video_path and os.path.exists(video_data.video_path):
            os.remove(video_data.video_path)

        file_path = os.path.join(OUTPUT_PATH_VIDEOS, f"{video_id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)

        return {"message": "video deleted"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




def launch():
    uvicorn.run("prediction:app", reload=True)

if __name__ == "__main__":
    launch()