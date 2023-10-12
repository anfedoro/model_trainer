import torch
import os

class TrainingStats:
    """
    A class to keep track of training statistics.

    Attributes:
    - train_losses (list): a list of training losses
    - val_losses (list): a list of validation losses
    - best_val_loss (float): the best validation loss so far
    - best_state_dict (dict): the state dictionary of the model with the best validation loss
    - epochs_trained (int): the number of epochs trained so far
    - learning_rates (list): a list of learning rates used during training
    """

    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_state_dict = None
        self.epochs_trained = 0
        self.learning_rates = []

    def load_from_checkpoint(self, checkpoint_path, device):
        """
        Load training statistics from a checkpoint file.

        Args:
        - checkpoint_path (str): the path to the checkpoint file
        - device (str): the device to load the checkpoint to
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_state_dict = checkpoint['best_state_dict']
        self.epochs_trained = checkpoint['epochs_trained']
        self.learning_rates = checkpoint['learning_rates']

    def save_to_checkpoint(self, checkpoint_path, model, optimizer, lr):
        """
        Save training statistics to a checkpoint file.

        Args:
        - checkpoint_path (str): the path to the checkpoint file
        - model (torch.nn.Module): the model being trained
        - optimizer (torch.optim.Optimizer): the optimizer used for training
        - lr (float): the current learning rate
        """
        checkpoint = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'best_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epochs_trained': self.epochs_trained + 1,
            'learning_rates': self.learning_rates + [lr]
        }
        torch.save(checkpoint, checkpoint_path)

    def update_train(self, loss):
        """
        Update the training loss.

        Args:
        - loss (float): the current training loss
        """
        self.train_losses.append(loss)

    def update_val(self, loss, model):
        """
        Update the validation loss and the best state dictionary.

        Args:
        - loss (float): the current validation loss
        - model (torch.nn.Module): the model being trained
        """
        self.val_losses.append(loss)
        if loss < self.best_val_loss:
            self.best_val_loss = loss
            self.best_state_dict = model.state_dict().copy()

    def get_best_state_dict(self):
        """
        Get the state dictionary of the model with the best validation loss.

        Returns:
        - best_state_dict (dict): the state dictionary of the model with the best validation loss
        """
        return self.best_state_dict


class ModelTrainer:
    """
    A class to train a PyTorch model.

    Attributes:
    - model (torch.nn.Module): the model to be trained
    - optimizer (torch.optim.Optimizer): the optimizer used for training
    - criterion (torch.nn.Module): the loss function used for training
    - device (str): the device used for training
    - stats (TrainingStats): the training statistics
    - save_best_only (bool): whether to save only the best model weights
    """

    def __init__(self, model, optimizer, criterion, device=None, continue_training=False, checkpoint_path=None, save_best_only=True):
        """
        Initialize the ModelTrainer.

        Args:
        - model (torch.nn.Module): the model to be trained
        - optimizer (torch.optim.Optimizer): the optimizer used for training
        - criterion (torch.nn.Module): the loss function used for training
        - device (str): the device used for training
        - continue_training (bool): whether to continue training from a checkpoint
        - checkpoint_path (str): the path to the checkpoint file
        - save_best_only (bool): whether to save only the best model weights
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.stats = TrainingStats()
        self.save_best_only = save_best_only

        if continue_training:
            if checkpoint_path is None:
                raise ValueError("checkpoint_path must be provided when continue_training is True")
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

            self.stats.load_from_checkpoint(checkpoint_path, self.device)
            self.model.load_state_dict(self.stats.best_state_dict)
            self.optimizer.load_state_dict(self.stats.checkpoint['optimizer_state_dict'])

            
    def train_one_epoch(self, train_loader):
        """
        Train the model for one epoch.

        Args:
        - train_loader (torch.utils.data.DataLoader): the data loader for training data
        """
        self.model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.stats.update_train(loss.item())

    def validate(self, val_loader):
        """
        Validate the model.

        Args:
        - val_loader (torch.utils.data.DataLoader): the data loader for validation data
        """
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        self.stats.update_val(avg_val_loss, self.model)

    def train(self, num_epochs, train_loader, val_loader):
        """
        Train the model for multiple epochs.

        Args:
        - num_epochs (int): the number of epochs to train for
        - train_loader (torch.utils.data.DataLoader): the data loader for training data
        - val_loader (torch.utils.data.DataLoader): the data loader for validation data
        """
        for epoch in range(num_epochs):
            self.train_one_epoch(train_loader)
            self.validate(val_loader)

            # Print training progress
            print(f"Epoch {epoch + 1}/{num_epochs}, ",end='')
            print(f"Train loss: {self.stats.train_losses[-1]:.6f}, ",end='')
            print(f"Validation loss: {self.stats.val_losses[-1]:.6f}")
            #print(f"Current learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            if not self.save_best_only:
                self.stats.save_to_checkpoint('path_to_save_all_epochs.pt', self.model, self.optimizer, lr=self.optimizer.param_groups[0]['lr'])


    def save_best_model_weights(self, path):
        """
        Save the weights of the model with the best validation loss.

        Args:
        - path (str): the path to save the model weights
        """
        best_state_dict = self.stats.get_best_state_dict()
        torch.save(best_state_dict, path)
    
    def save_checkpoint(self, path):
        """
        Save the training statistics to a checkpoint file.

        Args:
        - path (str): the path to the checkpoint file
        """
        current_lr = self.optimizer.param_groups[0]['lr']
        self.stats.save_to_checkpoint(path, self.model, self.optimizer, current_lr)