import matplotlib.pyplot as plt

def plot_loss_and_accuracy(train_losses, val_losses, train_accuracies, val_accuracies, epochs=5):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(r"/Users/linyinghsiao/Desktop/github專案/鑄造產品質量檢查/Loss.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(r"/Users/linyinghsiao/Desktop/github專案/鑄造產品質量檢查/Accuracy.png")
    plt.close()
