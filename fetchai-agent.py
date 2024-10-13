import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import lightning as L
from uagents import Agent, Context
alice = Agent(name="alice", seed="alice recovery phrase")



class DummyDataset(Dataset):
    """A dummy dataset that generates random images and labels for testing."""
    def __init__(self, num_samples=1000, image_size=(3, 224, 224)):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        """Returns the total number of samples."""
        return self.num_samples

    def __getitem__(self, idx):
        """Generates a random image and label."""
        image = torch.rand(self.image_size)  
        label = torch.randint(0, 2, (1,)).float().squeeze()  
        return image, label

class VGG16Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.vgg16(pretrained=True)
        self.model.classifier[6] = nn.Linear(4096, 1)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x).squeeze() 

    def compute_loss(self, logits, labels):
        """Compute the loss."""
        return self.criterion(logits, labels)  

class MyModel(L.LightningModule):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = VGG16Classifier()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)  
        loss = self.model.compute_loss(logits, y)  
        self.log('train_loss', loss) 
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x) 
        loss = self.model.compute_loss(logits, y) 
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x) 
        loss = self.model.compute_loss(logits, y)  
        self.log('test_loss', loss)  
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

if __name__ == "__main__":
    @alice.on_event("startup")
    async def introduce_agent(ctx: Context):
        ctx.logger.info(f"Hello, I'm agent {alice.name} and my address is {alice.address}.")
    alice.run()

    dataset = DummyDataset(num_samples=1000)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    model = MyModel()
    trainer = L.Trainer(max_epochs=5)

    trainer.fit(model, train_loader, val_loader)

    trainer.test(model, test_loader)
