import torch
import numpy as np
import skimage.metrics as metrics

def mse(imageA, imageB):
    return np.mean((imageA - imageB) ** 2)

def psnr(imageA, imageB):
    return metrics.peak_signal_noise_ratio(imageA, imageB)

def mae(imageA, imageB):
    return np.mean(np.abs(imageA - imageB))

def calculate_metrics(model, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    total_mse, total_psnr, total_mae = 0, 0, 0
    with torch.no_grad():
        for low_img, high_img in dataloader:
            low_img, high_img = low_img.to(device), high_img.to(device)
            outputs = outputs.cpu().numpy()
            
            for i in range(len(outputs)):
                total_mse += mse(outputs[i], high_img[i])
                total_psnr += psnr(outputs[i], high_img[i])
                total_mae += mae(outputs[i], high_img[i])
                
    num_images = len(dataloader.dataset)
    return total_mse / num_images, total_psnr / num_images, total_mae / num_images