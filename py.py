import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the CNN model
model = Sequential([
    Input(shape=(64, 64, 3)),  # Input shape of 64x64 with 3 color channels (RGB)
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# Load training data
training_set = train_datagen.flow_from_directory(
    'C:/Users/HP/OneDrive/Desktop/DL/water_bottle',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Train the model
model.fit(training_set, epochs=10, steps_per_epoch=len(training_set))


# Function to preprocess frames from the webcam
def preprocess_frame(frame):
    frame = cv2.resize(frame, (64, 64))
    frame = np.expand_dims(frame, axis=0)
    return frame / 255.0

# Start webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    preprocessed_frame = preprocess_frame(frame)
    prediction = model.predict(preprocessed_frame)

    # Determine the label based on prediction
    label = "Water Bottle" if prediction > 0.5 else "Not Water Bottle"

    # Display the label on the frame
    cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Live Detection', frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
