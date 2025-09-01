import torch
from loop_controller import SizedDataset
from typing import Any, Callable, Union
import matplotlib.pyplot as plt

def rand_img_grid(*, dataset: SizedDataset, cols: int, rows: int, title_from_label: Union[Callable[[Any], str], None] = None, figsize=(8, 8), cmap='gray'):
    figure = plt.figure(figsize=figsize)
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        title = title_from_label(label) if title_from_label else label
        plt.title(title)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap=cmap)
    plt.show()