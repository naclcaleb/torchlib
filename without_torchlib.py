import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision.transforms import v2

class_to_num = {
    'normal': 0,
    'pneum': 1
}

class PneumoniaDataset(Dataset):
    # Extra code to gather image files according to file structure
    def __init__(self, train=True, img_transform=None):
        self.image_dir = os.path.join('../input/chest-xray-images-guangzhou-women-and-childrens/chest_xray/', 'train' if train else 'test')
        self.lookup_table = []
        self.img_transform = img_transform

        # Load the table with labeled image paths
        for item in os.listdir(os.path.join(self.image_dir, 'NORMAL')):
            self.lookup_table.append((class_to_num['normal'], os.path.join(self.image_dir, 'NORMAL', item)))
        for item in os.listdir(os.path.join(self.image_dir, 'PNEUMONIA')):
            self.lookup_table.append((class_to_num['pneum'], os.path.join(self.image_dir, 'PNEUMONIA', item)))

        # Then, randomize it
        np.random.shuffle(self.lookup_table)

    def __len__(self):
        return len(self.lookup_table)

    def __getitem__(self, idx):
        label = self.lookup_table[idx][0]
        one_hot_label = torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(label), value=1)

        img_tensor = torchvision.io.decode_image(self.lookup_table[idx][1], mode=torchvision.io.ImageReadMode.GRAY)

        if self.img_transform:
            img_tensor = self.img_transform(img_tensor)

        return img_tensor, one_hot_label

train_dataset = PneumoniaDataset(train=True, img_transform=v2.Compose([
    v2.ToTensor(),
    v2.Resize((300, 300))
]))

labels_map = {
    0: "Normal",
    1: "Pneum"
}

# Code to create grid visualization
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
    img, label = train_dataset[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[int(torch.argmax(label).item())])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

class Classifier(nn.Module):
    def __init__(self, input_size=300, input_channels=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 10, 5),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 10, 5),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            # Manually calculating the linear input features,
            # Would have to be completely re-done if the input size was changed
            # or 2d layer hierarchy was adjusted
            nn.Linear(10 * 292 * 292, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x.float() / 255.0
        x = self.network(x)
        return x

model = Classifier()

learning_rate = 0.01
epochs = 200
batch_size = 40

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)

# Extra code for the training loop
# This example doesn't even have a test loop
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model.train()
for epoch in range(epochs):
    for batch, (imgs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        prediction = model(imgs)

        loss = loss_fn(prediction, labels)
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            print('Current loss: ', loss.item())