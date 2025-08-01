import torch

class NeuralNetwork:
    _device: str

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def run(self):
        pass
    
    def _train_model(self):
        pass
    
    def _accuracy_evaluation(self):
        pass
        