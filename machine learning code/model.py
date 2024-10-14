import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import lightning as L

class DummyDataset(Dataset):
    """A dummy dataset that generates random images and labels for testing."""
    def __init__(self, num_samples=100, image_size=(3, 224, 224)):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        """Returns the total number of samples."""
        return self.num_samples

    def __getitem__(self, idx):
        """Generates a random image and label."""
        # Generate a random image tensor
        image = torch.rand(self.image_size)  # Random image
        label = torch.randint(0, 2, (1,)).float().squeeze()  # Random label (0 or 1), squeeze to shape (1,)
        return image, label

class VGG16Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.vgg16(pretrained=True)
        self.model.classifier[6] = nn.Linear(4096, 1)  # Binary classification
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x).squeeze()  # Squeeze to shape (batch_size,)

    def compute_loss(self, logits, labels):
        """Compute the loss."""
        return self.criterion(logits, labels)  # Ensure y is of shape (batch_size,)

class MyModel(L.LightningModule):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = VGG16Classifier()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)  # Output shape: (batch_size,)
        loss = self.model.compute_loss(logits, y)  # Compute loss
        self.log('train_loss', loss)  # Log the training loss
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)  # Output shape: (batch_size,)
        loss = self.model.compute_loss(logits, y)  # Compute loss
        self.log('val_loss', loss)  # Log the validation loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)  # Output shape: (batch_size,)
        loss = self.model.compute_loss(logits, y)  # Compute loss
        self.log('test_loss', loss)  # Log the test loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

if __name__ == "__main__":
    # Create a dummy dataset
    dataset = DummyDataset(num_samples=1000)

    # Split the dataset into training, validation, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Instantiate and train the model
    model = MyModel()
    trainer = L.Trainer(max_epochs=5)

    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    # Test the model
    trainer.test(model, test_loader)
