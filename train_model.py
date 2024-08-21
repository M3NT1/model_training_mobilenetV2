import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint, Callback
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import matplotlib.pyplot as plt


class PerformanceCallback(Callback):
    def __init__(self):
        super().__init__()
        self.prev_train_acc = 0
        self.prev_train_loss = float('inf')
        self.prev_val_acc = 0
        self.prev_val_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        train_acc = logs.get('accuracy')
        train_loss = logs.get('loss')
        val_acc = logs.get('val_accuracy')
        val_loss = logs.get('val_loss')

        print(f"\nEpoch {epoch + 1} összegzés:")
        print(f"Train Accuracy: {train_acc:.4f} ({'javult' if train_acc > self.prev_train_acc else 'romlott'})")
        print(f"Train Loss: {train_loss:.4f} ({'javult' if train_loss < self.prev_train_loss else 'romlott'})")
        if val_acc is not None and val_loss is not None:
            print(f"Validation Accuracy: {val_acc:.4f} ({'javult' if val_acc > self.prev_val_acc else 'romlott'})")
            print(f"Validation Loss: {val_loss:.4f} ({'javult' if val_loss < self.prev_val_loss else 'romlott'})")

        self.prev_train_acc = train_acc
        self.prev_train_loss = train_loss
        self.prev_val_acc = val_acc if val_acc is not None else self.prev_val_acc
        self.prev_val_loss = val_loss if val_loss is not None else self.prev_val_loss


def create_model(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    # Finomhangolás: az utolsó 20 réteg taníthatóvá tétele
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    return model


def train_model(data_dir, epochs=50, batch_size=64):
    print(f"Adatkönyvtár: {data_dir}")
    if not os.path.exists(data_dir):
        print(f"Hiba: A megadott könyvtár nem létezik: {data_dir}")
        return

    print(f"Könyvtár tartalma: {os.listdir(data_dir)}")

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42)

    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=True,
        seed=42)

    if len(train_generator.class_indices) == 0:
        print(f"Hiba: Nem találhatók képek a megadott könyvtárban: {data_dir}")
        print("Kérlek, ellenőrizd az elérési utat és a könyvtárstruktúrát.")
        return

    num_classes = len(train_generator.class_indices)
    model = create_model(num_classes)

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Osztályok kiegyensúlyozása
    class_weights = compute_class_weight('balanced', classes=np.unique(train_generator.classes),
                                         y=train_generator.classes)
    class_weight_dict = dict(enumerate(class_weights))

    # Callbacks
    performance_callback = PerformanceCallback()
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1),
        TensorBoard(log_dir='./logs', histogram_freq=1),
        ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', verbose=1),
        performance_callback
    ]

    # Osztálynevek kiírása
    class_names = list(train_generator.class_indices.keys())
    print("Felismert osztályok:", class_names)

    # Osztályok egyensúlyának ellenőrzése
    for class_name, count in zip(class_names, np.bincount(train_generator.classes)):
        print(f"{class_name}: {count} képek")

    steps_per_epoch = len(train_generator)
    validation_steps = len(validation_generator)

    print(f"Train steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")

    # Végtelenített generátorok létrehozása class weight-tel
    def train_generator_func():
        while True:
            for x, y in train_generator:
                sample_weights = np.array([class_weight_dict[np.argmax(label)] for label in y])
                yield x, y, sample_weights

    def validation_generator_func():
        while True:
            for x, y in validation_generator:
                yield x, y

    history = model.fit(
        train_generator_func(),
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator_func(),
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=callbacks)

    model.save('custom_model.keras')

    # Osztálynevek mentése
    with open('class_names.txt', 'w') as f:
        f.write('\n'.join(class_names))

    print("Model saved as 'custom_model.keras'")
    print("Class names saved as 'class_names.txt'")

    # Modell értékelése a validációs halmazon
    evaluation = model.evaluate(validation_generator_func(), steps=validation_steps)
    print(f"Validation Loss: {evaluation[0]:.4f}")
    print(f"Validation Accuracy: {evaluation[1]:.4f}")

    # Tanulási görbék megjelenítése
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved as 'training_history.png'")


if __name__ == "__main__":
    # Használj abszolút elérési utat
    data_dir = "/Users/kasnyiklaszlo/PycharmProjects/model_training/data"
    train_model(data_dir)
