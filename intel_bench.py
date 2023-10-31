import torch
import torch.nn as nn
import torch.optim as optim
import intel_extension_for_pytorch as ipex
import time
import csv

torch.set_default_dtype(torch.float32)
# Define a simple neural network model with one hidden layer
class SimpleModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

input_size = 10
hidden_size = 500
output_size = 100
num_samples = 1000
# Generate random dummy data
data = torch.randn(num_samples, input_size, dtype=torch.float32)
# Random input data
target = torch.randn(num_samples, output_size, dtype=torch.float32)  # Random target values

train_dataset = torch.utils.data.TensorDataset(data, target)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize the model, criterion, and optimizer
model = SimpleModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training function
def train_one_epoch(
    train_loader: torch.utils.data.DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """
    Train the model for one epoch and return the average loss.
    
    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        model (nn.Module): Neural network model.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        device (torch.device): Device to run the model on (e.g., "cuda" or "cpu").
    
    Returns:
        float: Average loss for the epoch.
    """
    
    model.train()  # Set model to training mode
    total_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()  # Zero out gradients
        outputs = model(data)  # Forward pass
        loss = criterion(outputs, target)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        
        total_loss += loss.item()
    
    average_loss = total_loss / len(train_loader)
    return average_loss


# Example usage:
device = 'xpu' 
model = model.to(device)
model, optimizer =  ipex.optimize(model, optimizer=optimizer)
criterion.to(device)

# Initialize CSV logging
csv_file = "training_log.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Loss", "Time (seconds)"])  # Write headers
    
    elapsed_times = []
    
    for epoch in range(100):  # Training for 10 epochs
        start_time = time.time()
        avg_loss = train_one_epoch(train_loader, model, criterion, optimizer, device)
        elapsed_time = time.time() - start_time
        elapsed_times.append(elapsed_time)
        
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Time: {elapsed_time:.2f} seconds")
        
        # Log to CSV
        writer.writerow([epoch+1, f"{avg_loss:.4f}", f"{elapsed_time:.2f}"])
    
    # Calculate average time and log to CSV
    average_time = sum(elapsed_times) / len(elapsed_times)
    writer.writerow(["Average", "-", f"{average_time:.2f}"])