import torch
import torch.optim as optim

# 配置参数
train_dir = "/Users/linyinghsiao/Desktop/github專案/鑄造產品質量檢查/data/casting_data/train"
test_dir = "/Users/linyinghsiao/Desktop/github專案/鑄造產品質量檢查/data/casting_data/test"
batch_size = 64
epochs = 5
learning_rate = 0.001
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def get_criterion():
    return torch.nn.CrossEntropyLoss()

def get_optimizer(model):
    return optim.Adam(model.parameters(), lr=learning_rate)
