import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from age_gender_dataset import AgeGenderDataset
from model import AgeGenderNet
from torchvision import transforms
from torch.utils.data import random_split


device=torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(device)

train_transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

dataset=AgeGenderDataset(
    root_dir="data/UTKFace/UTKFace",
    transform=train_transform
)

train_size=int(
    0.8*len(dataset)
    )

val_size=len(dataset)-train_size

train_dataset,val_dataset=random_split(
    dataset,
    [train_size,val_size]
)

train_loader=DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader=DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)

model=AgeGenderNet().to(device)

age_loss_fn=nn.CrossEntropyLoss()

gender_loss_fn=nn.CrossEntropyLoss()

optimizer=torch.optim.Adam(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=1e-5
)

scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    patience=2
)

epochs=3
best_val_loss = float("inf")
for epoch in range(epochs):
    
    model.train()
    
    epoch_loss=0
    
    for batch_idx,(images,age_labels,gender_labels) in enumerate(train_loader):
        images = images.to(device)

        age_labels = age_labels.to(device)

        gender_labels = gender_labels.to(device)
    
        optimizer.zero_grad()
    
        age_logits,gender_logits=model(images)
    
        age_loss=age_loss_fn(
            age_logits,
            age_labels
        )
    
        gender_loss=gender_loss_fn(
            gender_logits,
            gender_labels
        )
    
        loss=age_loss+gender_loss

        loss.backward()

        optimizer.step()
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_loss+=loss.item()
        if batch_idx % 50==0:
            print(
                f"Epoch {epoch+1} | "
                f" Batch {batch_idx}/{len(train_loader)} | "
                f"Loss={loss.item():.4f}"
            )
    epoch_loss/=len(train_loader)


    
    model.eval()
    
    val_loss=0

    correct_age=0

    correct_gender=0
    
    total=0

    with torch.no_grad():
        for images,age_labels,gender_labels in val_loader:
            images = images.to(device)

            age_labels = age_labels.to(device)

            gender_labels = gender_labels.to(device)
            age_logits,gender_logits=model(images)

            age_predict=torch.argmax(
                age_logits,
                dim=1
            )
            gender_predict=torch.argmax(
                gender_logits,
                dim=1
            )

            correct_age+= (
                age_predict == age_labels
            ).sum().item()

            correct_gender+=(
                gender_predict==gender_labels
            ).sum().item()

            age_loss=age_loss_fn(
                age_logits,
                age_labels
            )

            gender_loss=gender_loss_fn(
                gender_logits,
                gender_labels
            )
            loss=age_loss+gender_loss

            val_loss+=loss.item()

            total+=age_labels.size(0)
        
        val_loss/=len(val_loader)
        scheduler.step(val_loss)
        age_accuracy=(
            100*correct_age/total
        )
        gender_accuracy=(
            100*correct_gender/total
        )
        if val_loss < best_val_loss:
            best_val_loss=val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss
                },
                "best_model.pth"
            )
        print(
            f"Epoch {epoch+1} | "
            f"Train Loss={epoch_loss:.4f} | "
            f"Val Loss={val_loss:.4f} | "
            f"Age Acc={age_accuracy:.2f}% | "
            f"Gender Acc={gender_accuracy:.2f}% | "
            f"LR={current_lr:.2f}"
        )

