import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from scripts.dataloader import CastingDataset
from scripts.det_cae import CAE
from scripts.det_vae import VAE
from scripts.det_attention_cae import AttentionAutoencoder

############################################################################
# 無監督或半監督方法
# 自編碼器（Autoencoder）：訓練一個自編碼器來重建無缺陷的圖像。
# 缺陷區域通常在重建誤差較高的地方，
# 可以通過計算圖像與重建圖像之間的差異來檢測缺陷區域。
# by using CAE VAE and attention-CAE
# https://doi.org/10.1016/j.ymssp.2021.108723
############################################################################

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)

# hyper parameter
config = {
    'seed': 111111,
    'valid_ratio':0.3,
    'EPOCH': 20,
    'BATCH_SIZE': 32,
    'LR': 0.0001,
    'kl_weight': 0.7,
    #'save_name': "CAE_8_13",
    #'model': CAE()
    #'save_name': "VAE_8_13",
    #'model': VAE()
    'save_name': "ATT_CAE_8_14",
    'model': AttentionAutoencoder()
}

train_ok_images_path = "data/casting_data/train/ok_front/"
test_ok_images_path = "data/casting_data/test/ok_front/"

train_dataset = CastingDataset(train_ok_images_path,test_ok_images_path)
train_loader = DataLoader(dataset=train_dataset, 
                               batch_size=32,
                               shuffle=True,
                               num_workers=0)

train_def_images_path = "data/casting_data/train/def_front/"
test_def_images_path = "data/casting_data/test/def_front/"

test_dataset = CastingDataset(train_def_images_path,test_def_images_path)
test_loader = DataLoader(dataset=train_dataset, 
                               batch_size=32,
                               shuffle=True,
                               num_workers=0)


model = config['model'].to(device)
#print(model)  # net architecture

# Construct the correct file path
model_file_name = f"{config['save_name']}.ckpt"
model_path = os.path.join("models", model_file_name)
# Load the model state dictionary
model.load_state_dict(torch.load(model_path,weights_only=True))

def detect_anomalies(model, image, threshold=0.3):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        output = model(image)
        output = output.cpu().squeeze(0)
        
        # Calculate reconstruction error
        reconstruction_error = torch.abs(image.cpu() - output)
        mask = reconstruction_error > threshold
        
        return reconstruction_error, mask

# Perform defect detection on a test image
test_img, _ = test_dataset[0]
reconstruction_error, mask = detect_anomalies(model, test_img)

# Display the defect detection results for the first 10 test images
num_examples = 5
plt.figure(figsize=(15, num_examples * 3))

for i in range(num_examples):
    test_img, _ = test_dataset[i]
    reconstruction_error, mask = detect_anomalies(model, test_img)
    
    plt.subplot(num_examples, 3, i * 3 + 1)
    plt.title("Original Image")
    plt.imshow(test_img.squeeze(), cmap='gray')
    
    plt.subplot(num_examples, 3, i * 3 + 2)
    plt.title("Reconstruction Error")
    plt.imshow(reconstruction_error.squeeze(), cmap='hot')
    
    plt.subplot(num_examples, 3, i * 3 + 3)
    plt.title("Defect Mask")
    plt.imshow(mask.squeeze(), cmap='gray')

plt.tight_layout()

pic_path = os.path.join("output", config['save_name'])
plt.savefig(pic_path)

plt.show()

