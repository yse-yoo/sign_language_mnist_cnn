import pandas as pd
import numpy as np
from PIL import Image
import os

# CSV 読み込み
train = pd.read_csv("data/sign_mnist_train.csv")

# 保存先フォルダ
os.makedirs("mnist_images", exist_ok=True)

count = 20  # 保存する画像の枚数
for i in range(count):
    label = train.iloc[i, 0]
    pixels = train.iloc[i, 1:].values.reshape(28,28).astype(np.uint8)
    img = Image.fromarray(pixels, mode='L')
    img.save(f"mnist_images/{label}_{i}.png")

print("画像を保存しました！")