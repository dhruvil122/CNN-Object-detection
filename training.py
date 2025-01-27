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

# Initialize the Model, Loss Function, and Optimizer
model = Network()
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001,weight_decay=1e-5)

# Load Data
train_loader = get_dataloader('/Users/dhruvil/Desktop/dl24/ex_christmas/data/train', batch_size=32, shuffle=True, training=True)
test_loader = get_dataloader('/Users/dhruvil/Desktop/dl24/ex_christmas/data/val', batch_size=32, shuffle=False, training=False)

# Train the Model
add_epochs = 10
initial_epochs = 0
total_epochs = initial_epochs + add_epochs
for epoch in range(initial_epochs, total_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch [{epoch+1}/{total_epochs}], Loss: {running_loss/len(train_loader):.4f}')

model.save_model()
print('Model is saved')
