#data_augmentation.py
import os
from torchvision import transforms
from PIL import Image

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class DataAugmentor:
    def __init__(self, input_dir, output_dir, augmentations, num_samples=1000):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.augmentations = transforms.Compose(augmentations)
        self.num_samples = num_samples
        create_dir(self.output_dir)

    def augment_data(self):
        for class_dir in ['def_front', 'ok_front']:
            class_input_dir = os.path.join(self.input_dir, class_dir)
            class_output_dir = os.path.join(self.output_dir, class_dir)
            create_dir(class_output_dir)
            images = [f for f in os.listdir(class_input_dir) if os.path.isfile(os.path.join(class_input_dir, f))]
            
            if len(images) == 0:
                print(f"No images found in {class_input_dir}")
                continue
            
            for i in range(self.num_samples):
                img_name = images[i % len(images)]
                image = Image.open(os.path.join(class_input_dir, img_name))
                augmented_image = self.augmentations(image)
                augmented_image.save(os.path.join(class_output_dir, f"augmented_{i}_{img_name}"))
                print(f"Saved augmented image: {os.path.join(class_output_dir, f'augmented_{i}_{img_name}')}")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    augmentations = [
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
    ]
    
    train_input_dir = os.path.join(base_dir, '../data/processed_data/train')
    train_output_dir = os.path.join(base_dir, '../data/augmentation/train')
    test_input_dir = os.path.join(base_dir, '../data/processed_data/test')
    test_output_dir = os.path.join(base_dir, '../data/augmentation/test')

    augmentor = DataAugmentor(train_input_dir, train_output_dir, augmentations)
    augmentor.augment_data()

    augmentor = DataAugmentor(test_input_dir, test_output_dir, augmentations)
    augmentor.augment_data()
