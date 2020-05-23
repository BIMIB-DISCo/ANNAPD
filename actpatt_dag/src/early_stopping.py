import torch
import numpy as np

class EarlyStopping():
    
    def __init__(self, model, val_set, patience, criterion):
        self.model = model
        self.patience = patience
        self.min_val_loss = np.inf
        self.val_set = val_set
        self.best_model = model
        self.no_improve = 0
        self.stop_crit = False
        
        self.criterion = criterion
        
    def step(self):
        
        if self.no_improve >= self.patience:
            raise Exception('Stopping criterion already met!')
            
        val_loss = 0
        
        for images, labels, _ in self.val_set:
            
            output = []
            with torch.no_grad():
                output = self.model(images)

            # Compute loss
            loss = self.criterion(output, labels)

            # Compute loss
            val_loss += loss.item()
        
        if val_loss <= self.min_val_loss + 1:
            if val_loss < self.min_val_loss:
                self.min_val_loss = val_loss
                self.best_model = self.model
            self.no_improve = 0
        else:
            self.no_improve += 1
        
        if self.no_improve == self.patience:
            self.stop_crit = True
    
    def stop(self):
        return self.stop_crit
    
    def get_best_model(self):
        return self.best_model