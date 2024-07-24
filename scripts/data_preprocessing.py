#data_preprocessing.py
import os
import cv2

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class DataPreprocessor:
    def __init__(self, input_dir, output_dir, size=(256, 256)):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.size = size
        create_dir(self.output_dir)

    def preprocess_image(self, image_path, output_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to read image {image_path}")
            return
        image = cv2.resize(image, self.size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(output_path, image)
        print(f"Processed {output_path}")

    def preprocess(self):
        for subdir, _, files in os.walk(self.input_dir):
            for file in files:
                input_path = os.path.join(subdir, file)
                output_subdir = subdir.replace(self.input_dir, self.output_dir)
                create_dir(output_subdir)
                output_path = os.path.join(output_subdir, file)
                self.preprocess_image(input_path, output_path)

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_input_dir = os.path.join(base_dir, '../data/casting_data/train')
    train_output_dir = os.path.join(base_dir, '../data/processed_data/train')
    test_input_dir = os.path.join(base_dir, '../data/casting_data/test')
    test_output_dir = os.path.join(base_dir, '../data/processed_data/test')

    preprocessor = DataPreprocessor(train_input_dir, train_output_dir)
    preprocessor.preprocess()

    preprocessor = DataPreprocessor(test_input_dir, test_output_dir)
    preprocessor.preprocess()
