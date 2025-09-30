from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()
model = tf.keras.models.load_model("asl_mnist_cnn.h5")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("L").resize((28,28))
    arr = np.array(image).reshape(1,28,28,1) / 255.0
    pred = model.predict(arr)
    return {"label": int(np.argmax(pred))}