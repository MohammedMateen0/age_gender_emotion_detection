# Real-Time Age, Gender & Emotion Detection using Deep Learning

## Project Overview

This project is an end-to-end Computer Vision application that performs:

* Face Detection
* Age Classification
* Gender Classification
* Emotion Recognition

The system supports both static image inference and real-time webcam prediction through an interactive Streamlit application.

The project demonstrates the complete Deep Learning workflow including dataset preprocessing, custom dataset creation, transfer learning, model fine-tuning, evaluation, inference pipelines, and deployment.

---

# Motivation

Human face analysis is one of the most widely used applications of Computer Vision. Many real-world systems require automatic extraction of facial attributes such as:

* Estimated age
* Gender
* Emotional state

Examples include:

* Smart surveillance systems
* Human-computer interaction
* Customer analytics
* Healthcare applications
* Retail analytics
* Interactive AI assistants

This project combines all three tasks into a single application.

---

# Features

## Face Detection

Faces are detected using OpenCV Haar Cascade Classifiers before classification.

## Age Classification

Predicts one of six age groups:

| Class | Age Group |
| ----- | --------- |
| 0     | 0-10      |
| 1     | 11-20     |
| 2     | 21-30     |
| 3     | 31-40     |
| 4     | 41-50     |
| 5     | 51+       |

---

## Gender Classification

Predicts:

* Male
* Female

---

## Emotion Recognition

Predicts seven facial emotions:

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprise

---

## Real-Time Webcam Inference

The webcam application:

* Detects faces in real time
* Crops face regions
* Performs Age prediction
* Performs Gender prediction
* Performs Emotion prediction
* Displays predictions directly on video frames

---

## Streamlit Web Application

Supports:

### Image Upload

Upload an image and receive:

* Age prediction
* Gender prediction
* Emotion prediction
* Prediction confidence scores

### Live Webcam

Perform real-time facial analysis directly from the browser.

---

# Datasets

## UTKFace Dataset

Used for:

* Age Classification
* Gender Classification

Dataset characteristics:

* 20,000+ facial images
* Large age diversity
* Balanced gender distribution
* Real-world facial variations

Filename format:

```text
age_gender_race_timestamp.jpg
```

Example:

```text
25_0_2_20170116174525125.jpg
```

Where:

* 25 = Age
* 0 = Male
* 2 = Race

---

## FER2013 Dataset

Used for:

* Emotion Recognition

Dataset characteristics:

* 35,000+ images
* 48x48 grayscale images
* 7 emotion classes

Dataset structure:

```text
FER2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
│
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

---

# Model Architecture

## Age & Gender Model

### Backbone

Pretrained ResNet18

```text
Input Image
↓
ResNet18 Backbone
↓
Feature Vector
├── Age Head
└── Gender Head
```

### Transfer Learning Strategy

* Pretrained ImageNet weights
* Freeze backbone layers
* Replace classification layer
* Fine-tune Layer4

### Output Heads

#### Age Head

```text
512 → 6
```

#### Gender Head

```text
512 → 2
```

---

## Emotion Model

### Backbone

Pretrained ResNet18

```text
Input Face
↓
ResNet18 Backbone
↓
Emotion Head
↓
7 Emotion Classes
```

### Fine-Tuning Strategy

* Freeze early layers
* Unfreeze Layer3
* Unfreeze Layer4
* Train custom emotion classifier

---

# Data Preprocessing

## Age & Gender Pipeline

```text
Image
↓
Resize (224×224)
↓
Random Horizontal Flip
↓
ToTensor
↓
Normalize
```

---

## Emotion Pipeline

```text
Grayscale Image
↓
Convert to 3 Channels
↓
Resize (224×224)
↓
Random Horizontal Flip
↓
Random Rotation
↓
Normalize
```

---

# Training Configuration

## Age & Gender

| Parameter     | Value            |
| ------------- | ---------------- |
| Optimizer     | Adam             |
| Learning Rate | 1e-5             |
| Batch Size    | 16               |
| Epochs        | 3                |
| Loss Function | CrossEntropyLoss |

---

## Emotion

| Parameter     | Value            |
| ------------- | ---------------- |
| Optimizer     | Adam             |
| Learning Rate | 1e-5             |
| Batch Size    | 32               |
| Epochs        | 10               |
| Loss Function | CrossEntropyLoss |

---

# Results

## Age Classification

Accuracy: **68%**

### Classification Summary

| Class | F1 Score |
| ----- | -------- |
| 0-10  | 0.90     |
| 11-20 | 0.44     |
| 21-30 | 0.75     |
| 31-40 | 0.41     |
| 41-50 | 0.28     |
| 51+   | 0.79     |

---

## Gender Classification

Accuracy: **92%**

### Classification Summary

| Class  | F1 Score |
| ------ | -------- |
| Male   | 0.92     |
| Female | 0.92     |

---

## Emotion Classification

Accuracy: **45%**

### Strongest Classes

| Emotion  | F1 Score |
| -------- | -------- |
| Happy    | 0.67     |
| Surprise | 0.62     |

### Challenging Classes

| Emotion | F1 Score |
| ------- | -------- |
| Fear    | 0.18     |
| Neutral | 0.14     |

---

# Project Structure

```text
age_gender_emotion_detection/

app/
├── streamlit_app.py
├── webcam_app.py

src/
├── age_gender_dataset.py
├── model.py
├── emotion_model.py
├── train_age_gender.py
├── train_emotion.py
├── predict.py
├── predict_emotion.py
├── evaluate.py

reports/
├── confusion_matrix_age.png
├── confusion_matrix_gender.png
├── confusion_matrix_emotion.png
├── classification_report.txt

README.md
requirements.txt
.gitignore
```

---

# Installation

```bash
git clone <repository-url>

cd age_gender_emotion_detection

pip install -r requirements.txt
```

---

# Training

## Train Age & Gender Model

```bash
python src/train_age_gender.py
```

## Train Emotion Model

```bash
python src/train_emotion.py
```

---

# Evaluation

```bash
python src/evaluate.py
```

Generated reports:

* Confusion Matrix (Age)
* Confusion Matrix (Gender)
* Confusion Matrix (Emotion)
* Classification Report

---

# Run Applications

## Image Upload App

```bash
streamlit run app/streamlit_app.py
```

## Webcam Application

```bash
streamlit run app/webcam_app.py
```

---

# Technologies Used

* Python
* PyTorch
* Torchvision
* OpenCV
* Streamlit
* Streamlit WebRTC
* NumPy
* Matplotlib
* Scikit-Learn
* PIL

---

# Future Improvements

* Multi-task unified model
* MTCNN face detection
* YOLO-based face detection
* Mobile deployment
* ONNX optimization
* TensorRT acceleration
* Cloud deployment
* Docker containerization

---

# Author

Mohammed Mateen

Machine Learning & Data Science Portfolio Project
