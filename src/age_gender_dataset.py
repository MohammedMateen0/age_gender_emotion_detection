from torch.utils.data import Dataset
from PIL import Image
import os

class AgeGenderDataset(Dataset):
    def __init__(
            self,
            root_dir,
            transform=None
    ):
        self.root_dir=root_dir
        self.transform=transform
        self.image=[
            img
            for img in os.listdir(self.root_dir)
            if img.endswith(".jpg")
        ]

    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, idx):
        filename=self.image[idx]
        parts=filename.split("_")
        age=int(parts[0])
        gender=int(parts[1])
        if age <= 10:
            age_group = 0

        elif age <= 20:
            age_group = 1

        elif age <= 30:
            age_group = 2

        elif age <= 40:
            age_group = 3

        elif age <= 50:
            age_group = 4

        else:
            age_group = 5
        image_path=os.path.join(
            self.root_dir,
            filename
        )
        image=Image.open(
            image_path
        ).convert("RGB")
        if self.transform:
            image=self.transform(image)
        return image,age_group,gender
    
