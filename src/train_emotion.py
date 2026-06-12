import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from emotion_model import EmotionNet
from collections import Counter

counter = Counter()
device=torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(device)

train_transform=transforms.Compose([
    transforms.Grayscale(
        num_output_channels=3
    ),
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.486],
        std=[0.229,0.224,0.225]
    )
])

test_transform = transforms.Compose([
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

train_dataset=ImageFolder(
    "data/FER2013/train",
    transform=train_transform
)
for _, label in train_dataset:
    counter[label] += 1

print("\nClass Distribution")
print(counter)
test_dataset=ImageFolder(
    "data/FER2013/test",
    transform=test_transform
)

print("\nClass Mapping")
print(train_dataset.class_to_idx)

train_loader=DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)
test_loader=DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

model=EmotionNet().to(device)
class_counts = torch.tensor(
    [3995, 436, 4097, 7215, 4965, 4830, 3171],
    dtype=torch.float
)

weights = (
    class_counts.sum()
    / class_counts
)

weights = weights.to(device)
loss_fn=nn.CrossEntropyLoss(weight=weights)

optimizer=torch.optim.AdamW(
    filter(
        lambda p:p.requires_grad,
        model.parameters()
    ),
    lr=1e-5,
    weight_decay=1e-4
)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    patience=2
)

epochs=10
best_val_loss=float("inf")

for epoch in range(epochs):
    model.train()

    train_loss=0
    for batch_idx,(images,labels) in enumerate(train_loader):
        
        images=images.to(device)
       
        labels=labels.to(device)
       
        optimizer.zero_grad()
       
        emotion_logits=model(images)

        loss=loss_fn(
            emotion_logits,
            labels
        )

        loss.backward()
        optimizer.step()

        train_loss+=loss.item()

        if batch_idx%50==0:
            print(
                f"Epoch: {epoch+1} | "
                f"Batch {batch_idx}/{len(train_loader)} | "
                f"Loss {loss.item():.4f}"
            )
    train_loss/=len(train_loader)
    model.eval()
    val_loss=0
    correct=0
    total=0
    with torch.no_grad():
        for images,labels in test_loader:
            images=images.to(device)
            labels=labels.to(device)

            emotion_logits=model(images)

            loss=loss_fn(
                emotion_logits,
                labels
            )

            val_loss+=loss.item()

            predictions=torch.argmax(
                emotion_logits,
                dim=1
            )
            correct+=(
                predictions==labels
            ).sum().item()

            total+=labels.size(0)
    val_loss/=len(test_loader)

    scheduler.step(val_loss)

    accuracy=(
        100*correct/total
    )
    current_lr=optimizer.param_groups[0]["lr"]

    if val_loss < best_val_loss:
        best_val_loss=val_loss
        torch.save(
            {
                "epoch":epoch,
                "model_state_dict":model.state_dict(),
                "optimizer_state_dict":optimizer.state_dict(),
                "val_loss":val_loss
            },
            "best_emotion_model.pth"
        )
    print(
        f"Epoch {epoch+1} | "
        f"Train Loss={train_loss:.4f} | "
        f"Val Loss={val_loss:.4f} | "
        f"Emotion Accuracy={accuracy:.4f}% | "
        f"LR={current_lr:.2e}"
    )