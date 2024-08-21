import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Konfiguráció
MODEL_PATH = 'custom_model.keras'
CLASS_NAMES_PATH = 'class_names.txt'
THRESHOLD = 0.5

# Osztálynevek betöltése
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

print(f"Betöltött osztályok: {class_names}")

# Modell betöltése
model = load_model(MODEL_PATH, compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Modell sikeresen betöltve")

# Kamera inicializálása
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Nem sikerült megnyitni a kamerát")
    exit()

print("Kamera sikeresen inicializálva")

while True:
    # Kép rögzítése a webkamerából
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Nem sikerült képkockát olvasni")
        continue

    # Kép előfeldolgozása
    resized_frame = cv2.resize(frame, (224, 224))
    preprocessed_frame = preprocess_input(np.expand_dims(resized_frame, axis=0))

    # Objektum felismerés
    predictions = model.predict(preprocessed_frame, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    # Eredmény megjelenítése
    if confidence > THRESHOLD:
        text = f"{class_names[predicted_class]}: {confidence:.2f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print(f"Felismert objektum: {text}")
    else:
        print(
            f"Nem található magas konfidenciájú felismerés. Legjobb találat: {class_names[predicted_class]}: {confidence:.2f}")

    # Összes osztály konfidenciájának kiírása
    for i, conf in enumerate(predictions[0]):
        print(f"{class_names[i]}: {conf:.2f}")

    # Kép megjelenítése
    cv2.imshow('Object Detection', frame)

    # Kilépés 'q' gomb megnyomására
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Erőforrások felszabadítása
cap.release()
cv2.destroyAllWindows()

print("Program vége")
