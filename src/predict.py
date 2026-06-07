from model import AgeGenderNet
import torch
from torchvision import transforms
from PIL import Image

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

device=torch.device(
    "cpu"
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

image=Image.open(
    "test_images/person.jpg"
).convert("RGB")

image=transform(image)

image=image.unsqueeze(0)
with torch.no_grad():

    age_logits, gender_logits = model(image)

    age_probs = torch.softmax(
        age_logits,
        dim=1
    )

    gender_probs = torch.softmax(
        gender_logits,
        dim=1
    )

    age_prediction = torch.argmax(
        age_probs,
        dim=1
    ).item()

    gender_prediction = torch.argmax(
        gender_probs,
        dim=1
    ).item()

age_label = AGE_CLASSES[
    age_prediction
]

gender_label = GENDER_CLASSES[
    gender_prediction
]
age_prob = torch.softmax(
    age_logits,
    dim=1
)

gender_prob = torch.softmax(
    gender_logits,
    dim=1
)

age_confidence = (
    age_prob.max().item() * 100
)

gender_confidence = (
    gender_prob.max().item() * 100
)
print(
    f"Age Group : {age_label}"
)

print(
    f"Age Confidence : "
    f"{age_confidence:.2f}%"
)

print(
    f"Gender : {gender_label}"
)

print(
    f"Gender Confidence : "
    f"{gender_confidence:.2f}%"
)
print("\nAge Probabilities")

for idx, prob in enumerate(
    age_probs[0]
):
    print(
        f"{AGE_CLASSES[idx]} : "
        f"{prob.item()*100:.2f}%"
    )

print("\nGender Probabilities")

for idx, prob in enumerate(
    gender_probs[0]
):
    print(
        f"{GENDER_CLASSES[idx]} : "
        f"{prob.item()*100:.2f}%"
    )