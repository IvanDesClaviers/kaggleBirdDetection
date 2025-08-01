import cv2
import torch

from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader

from .NeuralNetwork import NeuralNetwork
from .BirdModel import BirdDataset, BirdModel, ResNet

LEARNING_RATE = 0.1
N_CLASS = 50
N_EPOCHS = 20
BATCH_SIZE = 50
N_STEPS_PRINT_LOSS = 10
CSV_RELATIVE_PATH = 'datasets/train_anno.csv'
IMAGES_RELATIVE_DIR_PATH = 'datasets/images/'

MODEL_OUT_NAME = "model.pt"

class BirdNeuralNetwork(NeuralNetwork):
    def __init__(self):
        super(self)
    
    
    def _train_model(self, model: ResNet, optimizer,  train_dataloader: DataLoader, 
                     test_dataloader: DataLoader, n_epochs: int):
    
        running_loss = 0.
        last_loss = 0.
        loss_values = []
        
        for epoch in range(n_epochs):
            for i, data in enumerate(train_dataloader):
                X_batch, y_batch = data
                X_batch, y_batch = X_batch.float().to(self._device), y_batch.to(self._device)
        
                y_pred  = model(X_batch.to(self._device))
                loss = torch.nn.functional.cross_entropy(y_pred, y_batch, reduction="mean")
        
                loss.backward()
                optimizer.step()
                
                # More optimal than optimizer.zero_grad()
                for param in model.parameters():
                    param.grad = None 
        
                # Gather data and report
                running_loss += loss.item()
                if i % N_STEPS_PRINT_LOSS == N_STEPS_PRINT_LOSS-1:
                    last_loss = running_loss / N_STEPS_PRINT_LOSS # loss per batch
                    running_loss = 0.
                    loss_values.append(last_loss)
                    print('Epoch {}  batch {} loss: {}'.format(epoch + 1, i + 1, last_loss))
                    
            self._accuracy_evaluation(model, test_dataloader)
    
    @staticmethod
    def _accuracy_evaluation(self, model: ResNet, test_dataloader):
        acc = 0
        for j, data_test in enumerate(test_dataloader):
            X_test, y_test = data_test
            X_test, y_test = X_test.float().to(self._device), y_test.to(self._device)
             
            # For Some reason, all classes from the test csv are 0
            test_pred  = model(X_test.to(self._device))
             
            acc = acc + (torch.argmax(test_pred, dim=1) == y_test).float().mean()
             
        print('Accuracy: {}%'.format(acc/j)*100)
        print()  
        
        
    def PerformBirdDetection(self, csv :str = CSV_RELATIVE_PATH, 
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
            v2.RandomZoomOut(p=0.2),
            v2.RandomEqualize(p=0.1),
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
        
        model = BirdModel(n_class)
        optimizer = SGD(model.parameters(), lr=lr)
        model.train()
        
        self._train_model(model, optimizer,  train_dataloader, test_dataloader, n_epochs)
    
        torch.save(model.state_dict(), model_weights_name)
        print("Finished Training")
    
    
    
    
BirdNN = BirdNeuralNetwork()
BirdNN.run()