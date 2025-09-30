import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# データ読み込み
train = pd.read_csv("data/sign_mnist_train.csv")
test = pd.read_csv("data/sign_mnist_test.csv")

# 特徴量とラベルに分割
X_train = train.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255.0
y_train = train.iloc[:, 0].values

X_test = test.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255.0
y_test = test.iloc[:, 0].values

# ワンホットエンコーディング
num_classes = 25  # A〜Y, JとZを除外だがラベル番号は0〜24
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# 学習/検証に分割
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# CNN モデル構築
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 学習
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=128,
    validation_data=(X_val, y_val)
)

# 評価
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")

# モデル保存
model.save("asl_mnist_cnn")