import torch
import csv
from torch.utils.data import DataLoader
from data import ChristmasImages
from model import Network
import os

def get_dataloader(root_dir, batch_size=32, shuffle=True, training=True):
    dataset = ChristmasImages(root_dir, training=training)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# Initialize the Model
model = Network()  # Initialize your model class
model.load_state_dict(torch.load('/Users/dhruvil/Desktop/dl24/ex_christmas/model.pkl', map_location=torch.device('cpu')))  # Ensure model loads on CPU
model.eval()  # Set the model to evaluation mode

# Get the DataLoader for the test set
test_loader = get_dataloader('/Users/dhruvil/Desktop/dl24/ex_christmas/data/val/', batch_size=32, shuffle=False, training=False)

# List all image paths
image_paths = [os.path.join('/Users/dhruvil/Desktop/dl24/ex_christmas/data/val/', img) for img in os.listdir('/Users/dhruvil/Desktop/dl24/ex_christmas/data/val/')]

# Test the Model and collect predictions
predictions = []
with torch.no_grad():
    for i, inputs in enumerate(test_loader):
        if isinstance(inputs, list):  # Ensure 'inputs' is a Tensor
            inputs = inputs[0]
        inputs = inputs.to('cpu')  # Move to CPU explicitly
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        
        # Collect the predictions with image IDs
        for j, pred in enumerate(predicted):
            img_path = image_paths[i * test_loader.batch_size + j]
            image_id = os.path.splitext(os.path.basename(img_path))[0]  # Extract image ID
            predictions.append((int(image_id), int(pred.item())))

# Write predictions to a CSV file
csv_file_path = '/Users/dhruvil/Desktop/dl24/ex_christmas/validation_results.csv'
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Id', 'Category'])
    for image_id, prediction in predictions:
        writer.writerow([image_id, prediction])

print(f'Predictions saved to {csv_file_path}')
