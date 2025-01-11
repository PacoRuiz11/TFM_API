from ultralytics import YOLO
from fastapi import FastAPI
import uvicorn
from routers import prediction
from routers.prediction import OUTPUT_PATH_IMAGES, OUTPUT_PATH_VIDEOS
import shutil
import os
from contextlib import asynccontextmanager

#Function for delete the output directory
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting")
    yield
    print("Stopping")
    try:
        if os.path.exists(OUTPUT_PATH_IMAGES):
            shutil.rmtree(OUTPUT_PATH_IMAGES)
        if os.path.exists(OUTPUT_PATH_VIDEOS):
            shutil.rmtree(OUTPUT_PATH_VIDEOS)
    except Exception as e:
        print(f"Error deleting output directory: {str(e)}")


app = FastAPI(title="FastAPI with YOLOv8", lifespan=lifespan)

#routers
app.include_router(prediction.router)



@app.get("/")
async def read_root():
    return {"Hello": "World"}



def launch():
    uvicorn.run("main:app", reload=True, port=8080)    

if __name__ == "__main__":
    launch()