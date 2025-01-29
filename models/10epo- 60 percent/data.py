from torch.utils.data import Dataset
import torch
import os
from torchvision import transforms, datasets
from PIL import Image


class ChristmasImages(Dataset):
    
    def __init__(self, path, training=True):
        super().__init__()
        self.training = training
        self.path = path
        self.transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        if self.training == True:
            transforms.RandomResizedCrop(128)
            self.dataset = datasets.ImageFolder(root = self.path,transform = self.transform)
        else:
            transforms.Resize((128, 128))
            self.images = [os.path.join(path,img) for img in os.listdir(path)]
    
    def __len__(self):
        if self.training:
            return len(self.dataset)
        else:
            return len(self.images)
        # If training == True, path contains subfolders
        # containing images of the corresponding classes
        # If training == False, path directly contains
        # the test images for testing the classifier
        
    def __getitem__(self, index):
        if self.training == True:
            return self.dataset[index]
        else:
            image_path =self.images[index]
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
         
            return (image, )
