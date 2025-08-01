import torch
import torchvision
import torch.cuda as T
import torch.nn as nn
from torch.utils.data import DataLoader

N_STEPS_PRINT_LOSS = 10


class NeuralNetwork(nn.Module):
    _device: str

    def __init__(self):
        super().__init__()
        self._device = "cuda" if T.is_available() else "cpu"
    
    def _accuracy_evaluation(self) -> None: ...

    def _forward(self, net):
        return self.network( net )
    
    def train_model(self) -> None: ...
    


class BirdNeuralNetwork(NeuralNetwork):
    def __init__(self, num_classes: int, step_print_loss:int = N_STEPS_PRINT_LOSS):
        super().__init__()
        
        # We want to use ResNet to help differenciating birds, 
        # so we link the output of ResNet to a simple new classifyer
        self.network = torchvision.models.resnet34(weights='IMAGENET1K_V1').to(self._device)
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes).to(self._device) 
        self.step_print_loss = step_print_loss
    
    def train_model(self, optimizer,  train_dataloader: DataLoader, 
                     test_dataloader: DataLoader, n_epochs: int):

        running_loss = 0.
        last_loss = 0.
        loss_values = []
        
        for epoch in range(n_epochs):
            for i, data in enumerate(train_dataloader):
                X_batch, y_batch = data
                X_batch, y_batch = X_batch.float().to(self._device), y_batch.to(self._device)
        
                y_pred  = self._forward(X_batch.to(self._device))
                loss = torch.nn.functional.cross_entropy(y_pred, y_batch, reduction="mean")
        
                loss.backward()
                optimizer.step()
                
                # More optimal than optimizer.zero_grad()
                for param in self.parameters():
                    param.grad = None 
        
                # Gather data and report
                running_loss += loss.item()
                if i % self.step_print_loss == self.step_print_loss - 1:
                    last_loss = running_loss / N_STEPS_PRINT_LOSS # loss per batch
                    running_loss = 0.
                    loss_values.append(last_loss)
                    print('Epoch {}  batch {} loss: {}'.format(epoch + 1, i + 1, last_loss))
                    
            self._accuracy_evaluation(test_dataloader)
    
    def _accuracy_evaluation(self, test_dataloader):
        acc = 0
        for j, data_test in enumerate(test_dataloader):
            X_test, y_test = data_test
            X_test, y_test = X_test.float().to(self._device), y_test.to(self._device)
             
            # For Some reason, all classes from the test csv are 0
            test_pred  = self._forward(X_test.to(self._device))
             
            acc = acc + (torch.argmax(test_pred, dim=1) == y_test).float().mean()
             
        print('Accuracy: {}%'.format(acc/j)*100)
        print()  
    
    def _forward(self, net):
        return self.network( net )