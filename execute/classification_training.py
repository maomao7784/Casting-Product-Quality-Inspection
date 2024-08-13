import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from scripts.dataloader import CastingDataset
from scripts.cls_cnn import CNN
from scripts.cls_resnet import ResNet, block
# python -m execute.training

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)

# hyper parameter
config = {
    'seed': 111111,
    'valid_ratio':0.3,
    'EPOCH': 10,
    'BATCH_SIZE': 32,
    'LR': 0.0001,
    'save_name': "CNN_8_14",
    'model': CNN()
    #'save_name': "ResNet50_8_14",
    #'model': ResNet(block, [3, 4, 6, 3], 1, 2)
}

# load data
train_def_images_path = "data/casting_data/train/def_front/"
train_ok_images_path = "data/casting_data/train/ok_front/"

dataset = CastingDataset(train_def_images_path, train_ok_images_path)

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

optimizer = torch.optim.Adam(model.parameters(), lr=config['LR'])   # optimize all cnn parameters
criterion = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# initialize TensorBoard Summary
sumarry_folder = f"runs/{config['save_name']}"
writer = SummaryWriter(sumarry_folder)

# training and testing
for epoch in range(config['EPOCH']  ):
    model.train() # Set your model to train mode.
    # These are used to record information in training.
    train_loss = []
    train_accs = []

    for x, y in train_loader:  # gives batch data, normalize x when iterate train_loader
        b_x = Variable(x).to(device) # batch x
        b_y = Variable(y).to(device) # batch y
        # print(b_x.size())

        output = model(b_x)                   # cnn output
        loss = criterion(output, b_y)       # cross entropy loss
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients

        # Compute the accuracy for current batch.
        acc = (output.argmax(dim=-1) == b_y).float().mean()

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)
    print(f'Epoch: [{epoch+1}/{config['EPOCH']}] | train loss: {train_loss:.4f} | train accuracy: {train_acc:.4f}')

    # Each epoch record loss and acc
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)


    # Visualize model's weight and grad
    #for name, param in cnn.named_parameters():
    #    writer.add_histogram(name, param, epoch)
    #    writer.add_histogram(f'{name}.grad', param.grad, epoch)

    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval() # Set your model to evaluation mode.
    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []

    for x, y in valid_loader:
        b_x = Variable(x).to(device) # batch x
        b_y = Variable(y).to(device) # batch y
    
        with torch.no_grad():
            pred = model(b_x)
            loss = criterion(pred, b_y)
        
        # Compute the accuracy for current batch.
        acc = (pred.argmax(dim=-1) == b_y).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)
    print(f'Epoch: [{epoch+1}/{config['EPOCH']}] | valid loss: {valid_loss:.4f} | valid accuracy: {valid_acc:.4f}')

    # Each epoch record loss and acc
    writer.add_scalar('Loss/valid', valid_loss, epoch)
    writer.add_scalar('Accuracy/valid', valid_acc, epoch)

# close SummaryWriter
writer.close()

# Construct the correct file path
model_file_name = f"{config['save_name']}.ckpt"
model_path = os.path.join("models", model_file_name)
# Save model
torch.save(model.state_dict(), model_path)
