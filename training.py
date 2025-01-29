import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import ChristmasImages

from model import Network

def get_dataloader(root_dir, batch_size=32, shuffle=True, training=True):
    dataset = ChristmasImages(root_dir, training=training)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def train(num_epochs=10,
          train_dir='/Users/dhruvil/Desktop/dl24/ex_christmas/data/train/',
          num_classes=8,
          batch_size=32,
          learning_rate=1e-3,
          weight_decay=1e-5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create data loaders
    train_loader = get_dataloader(train_dir, batch_size=batch_size, shuffle=True, training=True)
  
    # If your validation data is also organized in subfolders by class, set training=True
    # so it uses ImageFolder. If not, adjust accordingly.

    # Initialize model, loss, optimizer
    pre_resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model = Network(num_classes=num_classes).to(device)

# The 'pre_resnet' is a model. It has its own state_dict.
# So we do NOT call torch.load on it; instead, we copy its weights.
    model.load_state_dict(pre_resnet.state_dict(), strict=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training loop
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / total_train
        train_acc = 100.0 * correct_train / total_train

        print('training loss :' , train_loss)
        print('training acc:', train_acc)

    model.save_model()

if __name__ == "__main__":
    # Example usage:
    train(
        num_epochs=10,
        train_dir='/Users/dhruvil/Desktop/dl24/ex_christmas/data/train/',
        num_classes=8,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-5
    )
