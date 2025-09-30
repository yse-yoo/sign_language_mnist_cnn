import tensorflow as tf
import numpy as np
from PIL import Image
import os
import io
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# FastAPI 初期化
app = FastAPI()

# モデル読み込み
model = tf.keras.models.load_model("asl_mnist_cnn.h5")

# ラベルマップ
label_map = {
    0:"A", 1:"B", 2:"C", 3:"D", 4:"E",
    5:"F", 6:"G", 7:"H", 8:"I", 9:"K",
    10:"L", 11:"M", 12:"N", 13:"O", 14:"P",
    15:"Q", 16:"R", 17:"S", 18:"T", 19:"U",
    20:"V", 21:"W", 22:"X", 23:"Y"
}

# 推論関数
def predict_image(img_path):
    img = Image.open(img_path).convert("L").resize((28,28))
    arr = np.array(img).reshape(1,28,28,1) / 255.0
    pred = model.predict(arr, verbose=0)
    return np.argmax(pred)

def predict_image_bytes(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("L").resize((28,28))
    arr = np.array(img).reshape(1,28,28,1) / 255.0
    pred = model.predict(arr, verbose=0)
    return np.argmax(pred)

# 静的ファイル (mnist_images を公開)
app.mount("/mnist_images", StaticFiles(directory="mnist_images"), name="mnist_images")

# Jinja2 テンプレート設定
templates = Jinja2Templates(directory="templates")

@app.get("/images", response_class=HTMLResponse)
async def show_images(request: Request):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_folder = os.path.join(base_dir, "mnist_images")
    files = [f for f in os.listdir(image_folder) if f.lower().endswith((".png", ".jpg"))]

    results = []
    for f in files[:50]:  # 最初の50枚まで
        file_path = os.path.join(image_folder, f)
        label_index = predict_image(file_path)
        label = label_map.get(label_index, "?")
        results.append({"file": f, "pred": label})

    return templates.TemplateResponse(
        "images.html",
        {"request": request, "results": results}
    )

# アップロードページ
@app.get("/upload", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/predict_json")
async def predict_json(file: UploadFile = File(...)):
    contents = await file.read()
    label_index = predict_image_bytes(contents)
    label = label_map.get(label_index, "?")
    return {"filename": file.filename, "label_index": int(label_index), "label": label}