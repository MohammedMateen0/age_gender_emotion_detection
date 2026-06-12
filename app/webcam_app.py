import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import cv2
import av
import torch
import streamlit as st

from PIL import Image
from torchvision import transforms
from streamlit_webrtc import webrtc_streamer


from src.model import AgeGenderNet
from src.emotion_model import EmotionNet

AGE_CLASSES = {
    0: "0-10",
    1: "11-20",
    2: "21-30",
    3: "31-40",
    4: "41-50",
    5: "51+"
}

GENDER_CLASSES = {
    0: "Male",
    1: "Female"
}

EMOTIONS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise"
}

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

emotion_model = EmotionNet().to(device)

emotion_checkpoint = torch.load(
    "best_emotion_model.pth",
    map_location=device
)

emotion_model.load_state_dict(
    emotion_checkpoint["model_state_dict"]
)

emotion_model.eval()

model = AgeGenderNet().to(device)

checkpoint = torch.load(
    "best_model.pth",
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

st.title(
    "Real-Time Age & Gender Detection"
)


def video_frame_callback(frame):

    img = frame.to_ndarray(
        format="bgr24"
    )

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:

        face = img[
            y:y+h,
            x:x+w
        ]

        if face.size == 0:
            continue

        rgb_face = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        pil_face = Image.fromarray(
            rgb_face
        )

        tensor = transform(
            pil_face
        ).unsqueeze(0).to(device)

        with torch.no_grad():

            age_logits, gender_logits = model(
                tensor
            )

            emotion_logits = emotion_model(
                tensor
            )

            age_probs = torch.softmax(
                age_logits,
                dim=1
            )

            gender_probs = torch.softmax(
                gender_logits,
                dim=1
            )

            emotion_probs = torch.softmax(
                emotion_logits,
                dim=1
            )

            age_pred = torch.argmax(
                age_probs,
                dim=1
            ).item()

            gender_pred = torch.argmax(
                gender_probs,
                dim=1
            ).item()

            emotion_pred = torch.argmax(
                emotion_probs,
                dim=1
            ).item()

            age_conf = (
                age_probs.max().item()
                * 100
            )

            gender_conf = (
                gender_probs.max().item()
                * 100
            )

            emotion_conf = (
                emotion_probs.max().item()
                * 100
            )

        age_text = (
            f"Age: {AGE_CLASSES[age_pred]}"
            f" ({age_conf:.1f}%)"
        )

        gender_text = (
            f"Gender: {GENDER_CLASSES[gender_pred]}"
            f" ({gender_conf:.1f}%)"
        )

        emotion_text = (
            f"Emotion: {EMOTIONS[emotion_pred]}"
            f" ({emotion_conf:.1f}%)"
        )

        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            img,
            age_text,
            (x, y - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

        cv2.putText(
            img,
            gender_text,
            (x, y - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

        cv2.putText(
            img,
            emotion_text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

    return av.VideoFrame.from_ndarray(
        img,
        format="bgr24"
    )


webrtc_streamer(
    key="age-gender",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)