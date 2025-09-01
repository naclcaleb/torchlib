import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from typing import Callable, Any, cast, Union

class SizedDataset(Dataset[Any]):
    def __len__(self) -> int: ...

class TrainingLoopController:

    model: nn.Module
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    optimizer: torch.optim.Optimizer
    lr_scheduler: Union[torch.optim.lr_scheduler.LRScheduler, None]
    epochs: int
    batch_size: int
    train_dataset: SizedDataset
    test_dataset: SizedDataset
    validate_dataset: Union[SizedDataset, None]

    def __init__(self, *, model, loss_fn, optimizer, lr_scheduler=None, epochs, batch_size, train_dataset, test_dataset, validate_dataset=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        # Todo: implement lr_scheduler
        self.lr_scheduler = lr_scheduler 
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
        if validate_dataset:
            self.validate_dataloader = DataLoader(validate_dataset, batch_size=self.batch_size, shuffle=True)

    def run(self, progress_every=10, on_start_epoch=None, on_progress_update=None, accuracy_fn=None, on_validation_update=None, on_test_complete=None):
        for epoch in range(self.epochs):
            if on_start_epoch:
                on_start_epoch(epoch=epoch)
            self.model.train()
            for batch, (features, labels) in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                prediction = self.model(features)
                loss = self.loss_fn(prediction, labels)
                loss.backward()

                self.optimizer.step()

                if on_progress_update and batch % progress_every == 0:
                    on_progress_update(batch, loss.item())
            
            if not self.validate_dataloader:
                continue

            # Validation 
            self.model.eval()
            size = len(cast(SizedDataset, self.validate_dataloader.dataset))
            num_batches = len(self.validate_dataloader)
            correct = 0
            total_loss = 0
            with torch.no_grad():
                for features, labels in self.validate_dataloader:
                    predictions = self.model(features)
                    total_loss += self.loss_fn(predictions, labels).item()
                    if accuracy_fn:
                        correct += accuracy_fn(predictions, labels).type(torch.float).sum().item()
            avg_loss = total_loss / num_batches
            accuracy = correct / size
            if on_validation_update:
                on_validation_update(avg_loss=avg_loss, accuracy=accuracy)

        # Time for the test dataset
        self.model.eval()
        size = len(cast(SizedDataset, self.test_dataloader.dataset))
        num_batches = len(self.test_dataloader)
        correct = 0
        total_loss = 0
        with torch.no_grad():
            for features, labels in self.test_dataloader:
                predictions = self.model(features)
                total_loss += self.loss_fn(predictions, labels).item()
                if accuracy_fn:
                    correct += accuracy_fn(predictions, labels).type(torch.float).sum().item()
        avg_loss = total_loss / num_batches
        accuracy = correct / size if accuracy_fn else None

        if on_test_complete:
            on_test_complete(avg_loss=avg_loss, accuracy=accuracy)

