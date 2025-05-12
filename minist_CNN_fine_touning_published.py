import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random


"""
We fix the seed to ensure reproducibility.
"""
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


# ======== LOAD & NORMALIZE MNIST DATA ========
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]
isTounedModel = None

base_path = r"C:\Users\volam\Desktop\DataScience\04_data_science_language_and_tools\01 computer vision\touning_images"

# ======== FINE-TUNING IMAGES ========
# Fine-tuning does NOT mean comparing or matching images (e.g., via k-NN or hashes).
# It RE-trains the model slightly with a few custom images to help it generalize better to new styles.
# Think of it as: "Alongside the 60,000 examples you know, learn to recognize *these* styles too".
# Fine-tuning is performed separately from initial training to increase the impact of the new style.
# With only a few samples, it's better to fine-tune than to include them in the main training set.
# This avoids underfitting and ensures max attention to new writing styles.
def fine_touning(model):
    print("I do fine-tuning with custom images...")
    
    personal_images = []
    personal_labels = []

    file_map = {
        3: ["esempio_3.png", "esempio_3_1.png", "esempio_3_2.png"],
        4: ["esempio_4.png", "esempio_4_1.png", "esempio_4_2.png"],
        5: ["esempio_5.png", "esempio_5_1.png", "esempio_5_2.png"]
    }
    
    for label, file_list in file_map.items():
        for file_name in file_list:
            full_path = os.path.join(base_path, file_name)
            _, img = preprocess_image(full_path)
            personal_images.append(img)
            personal_labels.append(label)

        
    X_personal = np.vstack(personal_images)
    y_personal = to_categorical(personal_labels, num_classes=10)

    X_train_ft = np.concatenate((X_train, X_personal), axis=0)
    y_train_ft = np.concatenate((y_train, y_personal), axis=0)
    
    print("Start fine-tuning with custom images...")
    model.fit(X_train_ft, y_train_ft, validation_data=(X_test, y_test), epochs=15, batch_size=32, verbose=2)
    
    model.save(CNN_model_tuned_path)
    print("Fine-touning completato.")
    print("touning salvato.")
    return model



# ======== CNN MODEL ========
def CNN_model():
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

CNN_model_base_path = "mnist_cnn_model.h5"
CNN_model_tuned_path = "mnist_cnn_model_tuned.h5"

# ======== EVALUATION METRICS ========
def evaluate_and_plot(model):
    scores = model.evaluate(X_test, y_test, verbose=0)
    accuracy = scores[1] * 100
    error = 100 - accuracy
    print(f"\n📊 {title}Accuracy: {accuracy:.2f}% | Baseline Error: {error:.2f}%")

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(cmap='Blues')
    plt.title(f'Confusion matrix of {title}')
    plt.show()
    
    
    
# ======== TRAINING METRICS PLOTS ========
def draw_training_graphs(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy during training')
    plt.xlabel('Epoca')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss during training')
    plt.xlabel('Epoc')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    

# ======== EXTERNAL IMAGE PREPROCESSING ========
def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Impossibile leggere l'immagine da: {path}")
    
    img_resized = cv2.resize(img, (28, 28))
    img_inverted = cv2.bitwise_not(img_resized)
    img_normalized = img_inverted.astype('float32') / 255.0
    img_reshaped = img_normalized.reshape(1, 28, 28, 1)

    return img_resized, img_reshaped



# ======== TRAINING FUNCTION ========
def trainingModel():
    # Deletes previous models if any and retrains from scratch
    if os.path.exists(CNN_model_base_path):
        os.remove(CNN_model_base_path)
        print("Base model deleted")

    if os.path.exists(CNN_model_tuned_path):
        os.remove(CNN_model_tuned_path)
        print("Fine touned midel deleted")
    
    print("Proceeding with new training")

    model = CNN_model()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=128,
        verbose=2
    )
    model.save(CNN_model_base_path)
    print("Base model trained and saved")
    
    model = fine_touning(model)
    return model, history



# ======== PREDICT SINGLE IMAGE ========
def preditc(model):
    image_path = os.path.join(base_path, "test_4.png")
    try:
        img_vis, img_input = preprocess_image(image_path)
        prediction = model.predict(img_input)
        predicted_class = np.argmax(prediction)
        
        print(f"Predected class: {predicted_class}")

        plt.imshow(img_vis, cmap='gray')
        plt.title(f"Predicting of {title}: {predicted_class}")
        plt.axis('off')
        plt.show()

        print("Probability for each class:")
        for i, prob in enumerate(prediction[0]):
            print(f"Class {i}: {prob * 100:.2f}%")

        plt.figure(figsize=(8, 4))
        plt.bar(range(10), prediction[0])
        plt.xticks(range(10))
        plt.title("Probability for each class")
        plt.xlabel("Class")
        plt.ylabel("Probability")
        plt.show()

    except Exception as e:
        print(f"Error: {e}")


# ======== ENTRY POINT ========
model_base_exists = os.path.exists(CNN_model_base_path)
model_tuned_exists = os.path.exists(CNN_model_tuned_path)


if model_base_exists or model_tuned_exists:
    print("Saved template found. Choose the template to use:")
    print(" 0 - Retrain the base model from scratch and touning")
    print(" 1 - Load the base model")
    print(" 2 - Load the fine touned model")

    choose = input("choose: ")
    
    if choose == "0":
        model, history = trainingModel()
        isTounedModel = None
        
    elif choose == "1" and model_base_exists:
        model = load_model(CNN_model_base_path)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = None
        isTounedModel = False
        print("Base model loaded")
        
    elif choose == "2" and CNN_model_tuned_path:
        model = load_model(CNN_model_tuned_path)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = None
        isTounedModel = True
        print("Fine touning model loaded")
        
    else:
        print("Invalid choice")
        exit()
else:
    print("Model not found. Proceeding with training")
    model, history = trainingModel()

title = "Touned model" if isTounedModel else "Base model"


# ======== PLOT RESULTS IF AVAILABLE ========
if history is not None:
    draw_training_graphs(history)
    
else:
    print("No charts available")

print("== PREDICT ==")
preditc(model)
evaluate_and_plot(model)
