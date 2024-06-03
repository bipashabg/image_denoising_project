import os
import torch
from skimage import io
from data_loader import get_dataloader
from models import SimpleCNN

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load('model_epoch_10.pth'))
    
    test_loader = get_dataloader('data/test/low', 'data/test/low', batch_size=1, shuffle=False)
    
    model.eval()
    with torch.no_grad():
        for i, (low_img, _) in enumerate(test_loader):
            low_img = low_img.to(device)
            output = model(low_img)
            output = output.squeeze().cpu().numpy().transpose(1, 2, 0)
            io.imsave(f'data/test/predicted/{i+1}.png', output)

if __name__ == "__main__":
    main()
    