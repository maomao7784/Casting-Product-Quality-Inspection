import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
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

dataset = CastingDataset(train_ok_images_path,test_ok_images_path)

# Define the split ratio
train_size = int(config["valid_ratio"] * len(dataset))
val_size = len(dataset) - train_size

# Split the dataset
train_dataset, valid_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(dataset=train_dataset, 
                               batch_size=config['BATCH_SIZE'],
                               shuffle=True,
                               num_workers=0)
valid_loader = DataLoader(dataset=valid_dataset, 
                               batch_size=config['BATCH_SIZE'],
                               shuffle=True,
                               num_workers=0)

model = config['model'].to(device)
#print(model)  # net architecture

optimizer = torch.optim.Adam(model.parameters(), lr=config['LR'])
criterion = nn.MSELoss()

# initialize TensorBoard Summary
sumarry_folder = f"runs/{config['save_name']}"
writer = SummaryWriter(sumarry_folder)

for epoch in range(config['EPOCH']  ):
    model.train() # Set your model to train mode.
    # These are used to record information in training.
    train_loss = []

    for x, y in train_loader:
        img = Variable(x).to(device)
    
        if config['save_name'].split('_')[0] == "VAE":

            reconstructed_img = model(img)
            # MSE loss between original image and reconstructed one
            loss_mse =  ((img - reconstructed_img) ** 2).sum() / config["BATCH_SIZE"]
            # KL divergence between encoder distrib. and N(0,1) distrib. 
            loss_kl = model.encoder.kl / config["BATCH_SIZE"]
            # Get weighted loss
            loss = (loss_mse * (1 - config["kl_weight"]) 
                    + loss_kl * config["kl_weight"])
            
        else:
            output = model(img)
            loss = criterion(output, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record the loss and accuracy.
        train_loss.append(loss.item())

    train_loss = sum(train_loss) / len(train_loss)
    print(f'Epoch: [{epoch+1}/{config['EPOCH']}] | train loss: {train_loss:.4f}')

    # Each epoch record loss and acc
    writer.add_scalar('Loss/train', train_loss, epoch)

    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval() # Set your model to evaluation mode.
    # These are used to record information in validation.
    valid_loss = []

    for x, y in valid_loader:
        img = Variable(x).to(device) # batch x
    
        with torch.no_grad():
            if config['save_name'].split('_')[0] == "VAE":

                reconstructed_pred = model(img)
                # MSE loss between original image and reconstructed one
                loss_mse = ((img - reconstructed_pred) ** 2).sum() / config["BATCH_SIZE"]
                # KL divergence between encoder distrib. and N(0,1) distrib. 
                loss_kl = model.encoder.kl / config["BATCH_SIZE"]
                # Get weighted loss
                loss = (loss_mse * (1 - config["kl_weight"]) 
                        + loss_kl * config["kl_weight"])

            else:
                pred = model(img)
                loss = criterion(pred, img)
        
        # Record the loss and accuracy.
        valid_loss.append(loss.item())

    valid_loss = sum(valid_loss) / len(valid_loss)   
    print(f'Epoch: [{epoch+1}/{config['EPOCH']}] | valid loss: {valid_loss:.4f}')

    # Each epoch record loss and acc
    writer.add_scalar('Loss/valid', valid_loss, epoch)

# close SummaryWriter
writer.close()

# Construct the correct file path
model_file_name = f"{config['save_name']}.ckpt"
model_path = os.path.join("models", model_file_name)
# Save model
torch.save(model.state_dict(), model_path)