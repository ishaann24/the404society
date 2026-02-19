import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import random
from torchvision import transforms
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


CLASS_MAP = {
    100: 0,
    200: 1,
    300: 2,
    500: 3,
    550: 4,
    600: 5,
    700: 6,
    800: 7,
    7100: 8,
    10000: 9
}

NUM_CLASSES = 10


class OffroadDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)

    def encode_mask(self, mask):
        new_mask = np.zeros_like(mask)
        for k, v in CLASS_MAP.items():
            new_mask[mask == k] = v
        return new_mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, -1)

        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

       
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        if random.random() > 0.5:
            angle = random.randint(-10, 10)
            M = cv2.getRotationMatrix2D((128, 128), angle, 1)
            image = cv2.warpAffine(image, M, (256, 256))
            mask = cv2.warpAffine(mask, M, (256, 256), flags=cv2.INTER_NEAREST)
      

        mask = self.encode_mask(mask)

        image = transforms.ToTensor()(image)
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask


model = torchvision.models.segmentation.deeplabv3_resnet50(weights="DEFAULT")
model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
model = model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()



def compute_iou(pred, mask, num_classes=NUM_CLASSES):
    pred = torch.argmax(pred, dim=1)
    ious = []

    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (mask == cls)

        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        if union == 0:
            continue

        ious.append(intersection / union)

    return np.mean(ious)



dataset = OffroadDataset("dataset/images", "dataset/masks")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

EPOCHS = 40

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0
    total_iou = 0

    loop = tqdm(loader)

    for images, masks in loop:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)["out"]
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        iou = compute_iou(outputs, masks)

        total_loss += loss.item()
        total_iou += iou

        loop.set_postfix(loss=loss.item(), IoU=iou)

    print(f"\nEpoch {epoch+1}")
    print("Average Loss:", total_loss / len(loader))
    print("Mean IoU:", total_iou / len(loader))

torch.save(model.state_dict(), "model.pth")
print("Training complete. Model saved.")
