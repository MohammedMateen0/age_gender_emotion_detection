import torch
from PIL import Image
from torchvision import transforms
from emotion_model import EmotionNet

EMOTIONS={
    0:"angry",
    1:"disgust",
    2:"fear",
    3:"happy",
    4:"neuteral",
    5:"sad",
    6:"suprise"
}
device=torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
model=EmotionNet()

checkpoint=torch.load(
    "best_emotion_model.pth",
    map_location=device
)
model.load_state_dict(
    checkpoint["model_state_dict"]
)
model.eval()

transform=transforms.Compose([
    transforms.Grayscale(
        num_output_channels=3
    ),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])
image=Image.open(
    "test_images/person.jpg"
)
image=transform(image)

image=image.to(device)

image = image.unsqueeze(0)
with torch.no_grad():
    emotion_logits=model(image)

    emotion_probs=torch.softmax(
        emotion_logits,
        dim=1
    )
    emotion_prediction=torch.argmax(
        emotion_probs,
        dim=1
    ).item()
emotion_labels=EMOTIONS[emotion_prediction]

emotion_confidence=(
    emotion_probs.max().item()*100
)
print(
    f"Emotion: {emotion_labels}"
)
print(
    f"Confidence : "
    f"{emotion_confidence:.2f}%"
)

print(
    "\nEmotion Probabilities"
)

for idx, prob in enumerate(
    emotion_probs[0]
):
    print(
        f"{EMOTIONS[idx]} : "
        f"{prob.item()*100:.2f}%"
    )