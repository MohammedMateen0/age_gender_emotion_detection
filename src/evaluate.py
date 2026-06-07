import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from age_gender_dataset import AgeGenderDataset
from model import AgeGenderNet

device=torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])
dataset=AgeGenderDataset(
    root_dir="data/UTKFace/UTKFace",
    transform=transform
)
train_size=int(
    0.8*len(dataset)
)
val_size=len(dataset)-train_size
_,val_dataset=random_split(
    dataset,
    [train_size,val_size]
)

val_loader=DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)
model=AgeGenderNet().to(device)

checkpoint=torch.load(
    "best_model.pth",
    map_location=device
)
model.load_state_dict(
    checkpoint['model_state_dict']
)

model.eval()

age_true=[]
age_pred=[]
gender_true=[]
gender_pred=[]


with torch.no_grad():
    for images,age_label,gender_label in val_loader:
        images=images.to(device)
        age_logits,gender_logits=model(images)
        age_prediction=torch.argmax(
            age_logits,
            dim=1
        )
        gender_prediction=torch.argmax(
            gender_logits,
            dim=1
        )
        age_true.extend(
            age_label.numpy()
        )
        age_pred.extend(
            age_prediction.cpu().numpy()
        )
        gender_true.extend(
            gender_label.numpy()
        )
        gender_pred.extend(
            gender_prediction.cpu().numpy()
        )
age_cm=confusion_matrix(
    age_true,
    age_pred
)

disp=ConfusionMatrixDisplay(
    confusion_matrix=age_cm
)
disp.plot()

plt.savefig(
    "reports/confusion_matrix_age.png"
)
plt.close()

gender_cm=confusion_matrix(
    gender_true,
    gender_pred
)
disp=ConfusionMatrixDisplay(
    confusion_matrix=gender_cm
)
disp.plot()

plt.savefig(
    "reports/confusion_matrix_gender.png"
)

plt.close()

age_report = classification_report(
    age_true,
    age_pred
)

gender_report = classification_report(
    gender_true,
    gender_pred
)

with open(
    "reports/classification_report.txt",
    "w"
) as f:

    f.write(
        "AGE REPORT\n"
    )

    f.write(
        age_report
    )

    f.write(
        "\n\n"
    )

    f.write(
        "GENDER REPORT\n"
    )

    f.write(
        gender_report
    )

print("\nAGE REPORT\n")
print(age_report)

print("\nGENDER REPORT\n")
print(gender_report)