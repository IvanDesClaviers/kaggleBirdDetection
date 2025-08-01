import cv2
import torch


from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch.optim import SGD

from NeuralNetwork import BirdNeuralNetwork
from BirdDataset import BirdDataset

LEARNING_RATE = 0.1
N_CLASS = 50
N_EPOCHS = 20
BATCH_SIZE = 50
CSV_RELATIVE_PATH = 'datasets/train_anno.csv'
IMAGES_RELATIVE_DIR_PATH = 'datasets/images/'
MODEL_OUT_NAME = "model.pt"

  
def PerformBirdDetection(csv :str = CSV_RELATIVE_PATH, 
                         img_path :str = IMAGES_RELATIVE_DIR_PATH,
                         model_weights_name: str = MODEL_OUT_NAME,
                         n_class: int = N_CLASS, n_epochs: int = N_EPOCHS,
                         lr: float = LEARNING_RATE, batch_size:int = BATCH_SIZE) -> None:
    
    train_transforms = v2.Compose([
        v2.RandomRotation(degrees=30, interpolation=cv2.INTER_LINEAR),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.3),
        #v2.ColorJitter(brightness=(0.4,0.8 ), 
        #               contrast=(0.4, 0.8), 
        #               saturation=(0.4, 0.8), 
        #               hue=(-0.2, 0.4)),
        #v2.RandomPerspective(distortion_scale=0.6, p=0.2),
        #v2.RandomZoomOut(p=0.2),
       # v2.RandomEqualize(p=0.1),
        #v2.RandomAdjustSharpness(sharpness_factor=2, p=0.1),
        v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
        v2.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
            ) # Imagenet calculated values
    ])
    
    # Create model and datasets

    full_dataset = BirdDataset(csv, img_path, train_transforms)
    
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.8, 0.2])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    model = BirdNeuralNetwork(n_class)
    optimizer = SGD(model.parameters(), lr=lr)
    model.train()
    
    model.train_model(optimizer, train_dataloader, test_dataloader, n_epochs)

    torch.save(model.state_dict(), model_weights_name)
    print("Finished Training")

    

PerformBirdDetection()