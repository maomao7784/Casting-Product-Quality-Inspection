import torch
import torch.nn.functional as F
from scripts.config import get_criterion

criterion = get_criterion()

def validate(model, device, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()  # 累加损失
            pred = output.argmax(dim=1, keepdim=True)  # 取概率最大的类别
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / len(val_loader.dataset)
    print(f'\nValidation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)}'
          f' ({accuracy:.2f}%)\n')
    
    return val_loss, accuracy

def evaluate_model(model, device, data_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []
    all_images = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_images.extend(data.cpu().numpy())

    test_loss /= len(data_loader.dataset)
    accuracy = 100. * correct / len(data_loader.dataset)

    return test_loss, accuracy, all_preds, all_targets, all_images

def test(model, device, test_loader):
    test_loss, accuracy, _, _, _ = evaluate_model(model, device, test_loader, criterion)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%\n')
