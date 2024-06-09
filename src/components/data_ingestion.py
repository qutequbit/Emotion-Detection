import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_images_from_folder(folder):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        if not os.path.isdir(label_folder):
            continue
        for filename in os.listdir(label_folder):
            img_path = os.path.join(label_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
            if img is not None:
                img = cv2.resize(img, (48, 48))  # Resize to 48x48
                images.append(img)
                labels.append(label)  # Use string labels
    return images, labels

def preprocess_data(images, labels):
    images = np.array(images).reshape(-1, 48, 48, 1)  # Reshape for CNN input
    images = images / 255.0  # Normalize pixel values

    # Encode labels as integers
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    return images, labels

def load_dataset(dataset_dir):
    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')
    
    train_images, train_labels = load_images_from_folder(train_dir)
    test_images, test_labels = load_images_from_folder(test_dir)
    
    train_images, train_labels = preprocess_data(train_images, train_labels)
    test_images, test_labels = preprocess_data(test_images, test_labels)
    
    return train_images, train_labels, test_images, test_labels

if __name__ == "__main__":
    dataset_dir = 'C:\\Users\\mysel\\PROJECTS_GITHUB\\EmoDetect\\src\\data'
    train_images, train_labels, test_images, test_labels = load_dataset(dataset_dir)
    print(f"Loaded {len(train_images)} training images and {len(test_images)} test images.")
