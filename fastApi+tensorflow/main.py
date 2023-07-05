from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

POTATO_MODEL = tf.keras.models.load_model("../models/potatoes.h5")
# APPLE_MODEL = tf.keras.models.load_model("../working_cnn/models/appletrain.h5")
# CHERRY_MODEL = tf.keras.models.load_model("../working_cnn/models/cherrytrain.h5")
# CAPCICUM_MODEL = tf.keras.models.load_model("../working_cnn/models/capsicumTrain.h5")
# STRAWBERRY_MODEL = tf.keras.models.load_model("../working_cnn/models/strawberrytrain.h5")



origins = [
    "http://localhost:3000",  # Replace with your React app's URL
    # Add more origins as needed
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

async def returnDiseasesName(file, CLASS_NAMES, MODEL):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100*np.max(predictions[0]), 2)
    return predicted_class, confidence

@app.get("/ping")
async def ping():
    return "Hello, I am alive too "


@app.post("/potatoLeafPredict")
async def predict(file: UploadFile = File(...)):
    CLASS_NAMES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
    predicted_class, confidence= await returnDiseasesName(file, CLASS_NAMES, POTATO_MODEL)
    return {
        "class": predicted_class, 
        "confidence": confidence
        }

@app.post("/appleLeafPredict")
async def predict(file: UploadFile = File(...)):
    CLASS_NAMES = ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy"]
    predicted_class, confidence= await returnDiseasesName(file, CLASS_NAMES, APPLE_MODEL)
    return {
        "class": predicted_class, 
        "confidence": confidence
        }

@app.post("/cherryLeafPredict")
async def predict(file: UploadFile = File(...)):
    CLASS_NAMES = ["Cherry_Powdery_mildew", "Cherry_Healthy"]
    predicted_class, confidence= await returnDiseasesName(file, CLASS_NAMES, CHERRY_MODEL)
    return {
        "class": predicted_class, 
        "confidence": confidence
        }

@app.post("/capsicumLeafPredict")
async def predict(file: UploadFile = File(...)):
    CLASS_NAMES = ["Capsicum__Bacterial_spot", "Capsicum__Healthy"]
    predicted_class, confidence= await returnDiseasesName(file, CLASS_NAMES, CAPCICUM_MODEL)
    return {
        "class": predicted_class, 
        "confidence": confidence
        }

@app.post("/straberryLeafPredict")
async def predict(file: UploadFile = File(...)):
    CLASS_NAMES = ["Strawberry__Leaf_scorch", "Strawberry__healthy"]
    predicted_class, confidence= await returnDiseasesName(file, CLASS_NAMES, STRAWBERRY_MODEL)
    return {
        "class": predicted_class, 
        "confidence": confidence
        }



if __name__== "__main__":
    uvicorn.run(app, host='localhost', port=8000)