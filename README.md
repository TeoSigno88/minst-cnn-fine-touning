# MNIST CNN Fine-Tuning with Custom Handwritten Digits

This project demonstrates the training and fine-tuning of a Convolutional Neural Network (CNN) using the MNIST dataset. The model is further fine-tuned using a small set of custom handwritten digit images to help it generalize better to unseen handwriting styles.

---
## Project Structure

- `minist_CNN_fine_touning_published.py` – Main script for training, fine-tuning, evaluation, and prediction.
- `touning_images/` – Folder with custom handwritten digit images used for fine-tuning.
- `mnist_cnn_model.h5` – Base CNN model trained on MNIST.
- `mnist_cnn_model_tuned.h5` – Fine-tuned model saved after adapting to custom digits.
- `README.md` – This file.
---


---
## Features

**Base CNN training on MNIST**  
**Fine-tuning using a small set of custom digit samples**  
**Model saving/loading options**  
**Confusion matrix visualization**  
**Single image prediction and confidence display**  
**Training metrics visualization**

---



---
### Why Fine-Tuning Is Useful (Real-World Motivation)

The MNIST dataset contains 60,000 digit images (0–9) mostly written in **American handwriting styles**.  
As a result, a machine learning model trained purely on MNIST may struggle to accurately recognize digits written in other styles—for instance, the **European-style "4"**.

This project demonstrates how to **fine-tune the model with just a few additional images**, allowing it to generalize better to non-MNIST styles.

#### Advantage:
If you don't have access to thousands of extra samples to expand MNIST, fine-tuning is a powerful alternative.  
With only a **handful of new examples**, the model can learn to adapt and make accurate predictions on **new handwriting styles**, without full retraining.

- Fine-tuning does **not** compare or match images directly (like with k-NN or image hashes).  
- It **re-trains** the model slightly, helping it learn from your custom examples.  
- It's like saying: *"You already know 60,000 digits. Now learn these special ones too."*

This method avoids underfitting and ensures **maximum focus on the new styles**, which is especially useful when deploying models in real-world, varied environments.

---



---
### How it works

On the first run, the script will automatically train both the **base CNN model** and the **fine-tuned model**.

When prompted, choose one of the following options:

- Enter **`0`** to re-train both models from scratch  
- Enter **`1`** to load and test the **base CNN model only**  
- Enter **`2`** to load and test the **fine-tuned model**

After selecting your option, the script will display training graphs, confusion matrix, and prediction results.

---



## How to Run

### 1. Install Dependencies

Make sure you have Python 3.x installed. Then install the required packages:

```bash
pip install tensorflow matplotlib numpy opencv-python scikit-learn
```


## 🏷️ Keywords

fine-tuning CNN, MNIST handwritten digits, handwriting recognition, custom digit classification, deep learning, convolutional neural network, image classification, European digit style, transfer learning, Keras, TensorFlow
