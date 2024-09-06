import torch
import torch.optim as optim
from scripts.dataloader import get_data_loaders
from scripts.cls_cnn import CNN
from execute.classification_training import train
from execute.classification_testing import test, evaluate_model, validate
from scripts.gradcam import visualize_gradcam
from scripts.plot import plot_loss_and_accuracy
from scripts.config import train_dir, test_dir, batch_size, epochs, device, get_criterion, get_optimizer

# 加载数据
train_loader, val_loader, test_loader = get_data_loaders(train_dir, test_dir, batch_size)

# 初始化模型
model = CNN().to(device)

# 获取损失函数和优化器
criterion = get_criterion()
optimizer = get_optimizer(model)

# 训练和验证模型
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(1, epochs + 1):
    train_loss, train_accuracy = train(model, device, train_loader, optimizer, epoch, criterion)
    val_loss, val_accuracy = validate(model, device, val_loader)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

# 测试模型
test(model, device, test_loader)

# Grad-CAM 可视化
visualize_gradcam(model, device, test_loader, model.conv2, criterion)

# 绘制训练和验证的损失和准确度曲线
plot_loss_and_accuracy(train_losses, val_losses, train_accuracies, val_accuracies, epochs)

