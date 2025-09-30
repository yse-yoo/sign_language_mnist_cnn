import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# モデル読み込み
model = tf.keras.models.load_model("asl_mnist_cnn.h5")

# 数値ラベル → アルファベットの対応表
label_map = {
    0:"A", 1:"B", 2:"C", 3:"D", 4:"E",
    5:"F", 6:"G", 7:"H", 8:"I", 9:"K",
    10:"L", 11:"M", 12:"N", 13:"O", 14:"P",
    15:"Q", 16:"R", 17:"S", 18:"T", 19:"U",
    20:"V", 21:"W", 22:"X", 23:"Y"
}

# 画像を読み込んで推論する関数
def predict_image(img_path):
    img = Image.open(img_path).convert("L").resize((28,28))
    arr = np.array(img).reshape(1,28,28,1) / 255.0
    pred = model.predict(arr, verbose=0)
    return np.argmax(pred)

# mnist_images フォルダの全画像を取得
base_dir = os.path.dirname(os.path.abspath(__file__))
image_folder = os.path.join(base_dir, "mnist_images")
files = [f for f in os.listdir(image_folder) if f.lower().endswith((".png", ".jpg"))]

# matplotlib で一覧表示
plt.figure(figsize=(12, 8))
for i, file_name in enumerate(files[:20]):  # 最初の20枚を表示（数が多いと重いので制限）
    file_path = os.path.join(image_folder, file_name)
    label_index = predict_image(file_path)
    label = label_map.get(label_index, "?")
    
    img = Image.open(file_path)
    plt.subplot(4, 5, i+1)  # 4行5列で並べる
    plt.imshow(img, cmap="gray")
    plt.title(f"{file_name}\nPred: {label}")
    plt.axis("off")

plt.tight_layout()
plt.show()