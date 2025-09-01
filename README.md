# Example Usage
See `demo.py`. 

```python
from data.datasets import FileDataset, Matcher
from loop_controller import TrainingLoopController
import torch
import nn
import torchvision
from data import vis
from torchvision.transforms import v2

# Load datasets
dataset = FileDataset([ Matcher('chest_xray/{group}/{label}/{feature:full_path}') ])

label_map = { 'NORMAL': 0, 'PNEUMONIA': 1 }
l_transform = lambda label: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(label_map[label]), value=1)

def f_transform(file_path):
    img_tensor = torchvision.io.read_image(file_path, mode=torchvision.io.ImageReadMode.GRAY)
    # img_tensor = torchvision.io.decode_image(img_data, mode=torchvision.io.ImageReadMode.GRAY)
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((300, 300))
    ])
    return transform(img_tensor)

train_data = dataset.group('train', l_transform=l_transform, f_transform=f_transform)
test_data = dataset.group('test', l_transform=l_transform, f_transform=f_transform)

 
# Visualize the data
label_titles = ['Normal', 'Pneumonia']
vis.rand_img_grid(dataset=train_data, cols=4, rows=4, title_from_label=lambda label: label_titles[int(torch.argmax(label).item())])


# Create the model
class Classifier(torch.nn.Module):
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
            # Notice, after the Flatten layer we do not need to perform 
            # any calculations to get the input shape of the Linear layers.
            # However, in all other respects these modules work just like regular PyTorch modules
            nn.Linear(out_features=128),
            nn.ReLU(),
            nn.Linear(out_features=2),
            nn.Sigmoid(),
            input_shape=(input_size, input_size)
        )
    def forward(self, x):
        x = x.float() / 255.0
        x = self.network(x)
        return x

model = Classifier()

loop_controller = TrainingLoopController(
    model=model,
    loss_fn=torch.nn.MSELoss(),
    optimizer=torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5),
    epochs=200,
    batch_size=40,
    train_dataset=train_data,
    test_dataset=test_data
)

def on_start_epoch(epoch):
    print('Starting epoch: ' + epoch)
def on_progress_update(batch, loss):
    print('Batch number ' + batch + ', loss: ' + loss)

loop_controller.run(
    progress_every=20, 
    on_start_epoch=on_start_epoch,
    on_progress_update=on_progress_update
)
```