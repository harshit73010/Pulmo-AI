import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os

# -----------------------------
# Dataset Paths
# -----------------------------
train_dir = r"C:\Users\ASUS\Desktop\Disease_Detection\train"
test_image_path = r"C:\Users\ASUS\Downloads\archive (12)\chest_xray\test\PNEUMONIA\person33_virus_72.jpeg"

# -----------------------------
# Image Preprocessing
# -----------------------------
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode="binary"
)

# -----------------------------
# CNN Model
# -----------------------------
model = Sequential([
    Conv2D(32,(3,3),activation="relu",input_shape=(224,224,3)),
    MaxPooling2D(2,2),

    Conv2D(64,(3,3),activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128,(3,3),activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(128,activation="relu"),

    Dense(1,activation="sigmoid")
])

# -----------------------------
# Compile Model
# -----------------------------
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# Train Model
# -----------------------------
print("Training Model...")
model.fit(train_generator, epochs=10)

# -----------------------------
# Save Model
# -----------------------------
model.save("C:/Users/ASUS/Desktop/Disease_Detection/cnn_model.h5")
print("Model Saved Successfully!")

# -----------------------------
# Image Prediction
# -----------------------------
if os.path.exists(test_image_path):

    img = image.load_img(test_image_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array/255.0

    prediction = model.predict(img_array)

    if prediction > 0.5:
        print("Prediction: PNEUMONIA DETECTED")
    else:
        print("Prediction: NORMAL")

else:
    print("No test image found. Training completed.")

