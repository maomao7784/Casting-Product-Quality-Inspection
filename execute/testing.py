import os
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import pandas as pd
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from scripts.dataloader import CastingDataset
from scripts.nn_cnn import CNN
from scripts.nn_resnet import ResNet, block
# python -m execute.training

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

# hyper parameter
config = {
    'seed': 111111,
    'BATCH_SIZE': 32,
    'save_name': "cnn_8_9",
    'model': CNN()
    #'model': ResNet(block, [3, 4, 6, 3], 1, 2)
}

# load data
test_def_images_path = "data/casting_data/test/def_front/"
test_ok_images_path = "data/casting_data/test/ok_front/"

test_dataset = CastingDataset(test_def_images_path, test_ok_images_path)


test_loader = DataLoader(dataset=test_dataset, 
                               batch_size=config['BATCH_SIZE'],
                               shuffle=False,
                               num_workers=0)
test_target = test_dataset.labels

model = config['model'].to(device)

# Construct the correct file path
model_file_name = f"{config['save_name']}.ckpt"
model_path = os.path.join("models", model_file_name)
# Load the model state dictionary
model.load_state_dict(torch.load(model_path,weights_only=True))

model.eval()
# These are used to record information in test.
prediction = []
for x, y in test_loader:
    b_x = Variable(x).to(device) # batch x

    with torch.no_grad():
        test_pred = model(b_x)

    # save the prediction
    test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
    prediction += test_label.squeeze().tolist()
    

print("-------------------------------------test_mode-----------------------------------------")
# compute the acc of test.
test_acc = (np.array(prediction) == test_target).float().mean().numpy()
print('test accuracy: %.4f' % test_acc)

# create test csv
df = pd.DataFrame()
df["true"] = test_target
df["pred"] = prediction
df.to_csv("output/test_output.csv",index = False)