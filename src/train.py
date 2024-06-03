import torch
import torch.optim as optim
import torch.nn as nn
from data_loader import get_dataloader
from models import SimpleCNN

def train_model(train_loader, num_epochs=10, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for low_img, high_img in train_loader:
            low_img, high_img = low_img.to(device), high_img.to(device)
            
            optimizer.zero_grad()
            outputs = model(low_img)
            loss = criterion(outputs, high_img)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * low_img.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        
        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')
        
    return model

if __name__== "__main__":
    train_loader = get_dataloader('data/train/low', 'data/train/high')
    model = train_model(train_loader)