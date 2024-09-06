import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)  

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_image, class_idx=None):
        self.model.eval()
        output = self.model(input_image)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[:, class_idx].backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * activations, dim=0)

        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        cam = cam.cpu().numpy()

        cam = cv2.resize(cam, (input_image.size(3), input_image.size(2)))

        return cam

def visualize_gradcam(model, device, data_loader, target_layer, criterion, num_images=9):
    from execute.classification_testing import evaluate_model
    
    _, _, all_preds, all_targets, all_images = evaluate_model(model, device, data_loader, criterion)
    grad_cam = GradCAM(model, target_layer)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    for i, ax in enumerate(axes[:num_images]):
        img = all_images[i].transpose(1, 2, 0)
        img = (img * 0.5) + 0.5

        cam = grad_cam.generate(torch.tensor(all_images[i:i+1]).to(device))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        superimposed_img = heatmap + img
        superimposed_img = superimposed_img / np.max(superimposed_img)

        ax.imshow(superimposed_img)
        ax.set_title(f'Pred: {all_preds[i][0]}, True: {all_targets[i]}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(r"/Users/linyinghsiao/Desktop/github專案/鑄造產品質量檢查/GradCAM.png")
    plt.close()
