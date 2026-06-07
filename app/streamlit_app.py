import streamlit as st
import torch

from PIL import Image
from torchvision import transforms

from src.model import AgeGenderNet
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
AGE_CLASSES={
    0:"0-10",
    1:"11-20",
    2:"21-30",
    3:"31-40",
    4:"41-50",
    5:"51+"
}
GENDER_CLASSES={
    0:"Male",
    1:"Female"
}
device=torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model=AgeGenderNet()

checkpoint=torch.load(
    "best_model.pth",
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])
st.title(
    "Age & Gender Detection"
)

uploaded_file=st.file_uploader(
    "Upload Face Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:
    image=Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Upload Image"
    )

    x=transform(image)

    x=x.unsqueeze(0)

    with torch.no_grad():

        age_logits,gender_logits=model(x)

        age_probs=torch.softmax(
            age_logits,
            dim=1
        )
        gender_probs=torch.softmax(
            gender_logits,
            dim=1
        )
        age_prediction=torch.argmax(
            age_probs,
            dim=1
        ).item()

        gender_prediction=torch.argmax(
            gender_probs,
            dim=1
        ).item()
    
    st.subheader("Prediction")

    st.write(
        f"Age Group: "
        f"{AGE_CLASSES[age_prediction]}"
    )

    st.write(
        f"Gender: "
        f"{GENDER_CLASSES[gender_prediction]}"
    )

    st.subheader(
        "Age Probabilities"
    )

    age_dict={
        AGE_CLASSES[i]:
        age_probs[0][i].item()
        for i in range(6)
    }

    st.bar_chart(age_dict)

    st.subheader(
        "Gender Probabilities"
    )

    gender_dict={
        GENDER_CLASSES[i]:
        gender_probs[0][i].item()
        for i in range(2)
    }

    st.bar_chart(
        gender_dict
    )