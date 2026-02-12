import os
import shutil
import hashlib
import cv2
import numpy as np
import albumentations as A
import random
# from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

# Configuration
SOURCE_DIR = Path("data/raw/disease_detection/tomato_v2")
TARGET_DIR = Path("data/processed/tomato_disease")
TARGET_SIZE = (224, 224)  # Standard size for many models
SPLIT = {'train': 0.8, 'val': 0.1, 'test': 0.1}

# Mapping to consolidate classes
CLASS_MAPPING = {
    "leaf blight": "Leaf_blight",
    "septoria leaf spot": "Septoria_leaf_spot",
    "leaf curl": "Yellow_Leaf_Curl_Virus",
    "leaf mold": "Leaf_Mold",
    "healthy": "Healthy",
    "verticulium wilt": "Verticillium_wilt"
}

# Augmentation pipeline
aug_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussianBlur(p=0.1),
    A.GaussNoise(p=0.1),
])

def get_file_hash(filepath):
    """Calculate MD5 hash of a file to detect duplicates."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def process_dataset():
    print(f"Processing dataset from {SOURCE_DIR} to {TARGET_DIR}")
    
    if TARGET_DIR.exists():
        print(f"Cleaning target directory {TARGET_DIR}...")
        shutil.rmtree(TARGET_DIR)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    # Track file hashes to avoid duplicates
    seen_hashes = set()
    class_counts = {}
    
    # 1. Consolidate and Copy
    print("Consolidating and copying files...")
    source_folders = [f for f in SOURCE_DIR.iterdir() if f.is_dir()]
    
    for i, folder in enumerate(source_folders):
        print(f"Processing folder {i+1}/{len(source_folders)}: {folder.name}")
        folder_name = folder.name
        if folder_name not in CLASS_MAPPING:
            print(f"Warning: Skipping unknown folder '{folder_name}'")
            continue
            
        target_class = CLASS_MAPPING[folder_name]
        target_class_dir = TARGET_DIR / target_class
        target_class_dir.mkdir(exist_ok=True)
        
        for img_path in folder.glob("*"):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
            
            # Verify image is readable
            try:
                test_img = cv2.imread(str(img_path))
                if test_img is None:
                    print(f"Warning: Skipping corrupt image {img_path}")
                    continue
            except Exception as e:
                print(f"Warning: Error checking image {img_path}: {e}")
                continue

            file_hash = get_file_hash(img_path)
            if file_hash in seen_hashes:
                continue # Skip duplicate
            
            seen_hashes.add(file_hash)
            
            # Copy file
            shutil.copy2(img_path, target_class_dir / img_path.name)
            
            class_counts[target_class] = class_counts.get(target_class, 0) + 1

    print("\nClass distribution after consolidation:")
    max_count = 0
    for cls, count in class_counts.items():
        print(f"{cls}: {count}")
        max_count = max(max_count, count)
        
    # 2. Augment Healthy Class (and others to balance)
    print(f"\nTarget count per class: {max_count}")
    print("Augmenting 'Healthy' class to balance dataset...")
    
    # We can augment other classes too if desired, but user specified Healthy
    classes_to_augment = ["Healthy"] 
    
    for cls in classes_to_augment:
        if cls not in class_counts:
            print(f"Warning: Class {cls} not found in dataset.")
            continue
            
        current_count = class_counts[cls]
        needed = max_count - current_count
        
        if needed <= 0:
            print(f"Class {cls} already has enough images ({current_count}).")
            continue
            
        print(f"Augmenting {cls}: generating {needed} new images...")
        class_dir = TARGET_DIR / cls
        images = list(class_dir.glob("*"))
        
        generated_count = 0
        # pbar = tqdm(total=needed)
        
        while generated_count < needed:
            if generated_count % 100 == 0:
                print(f"  Generated {generated_count}/{needed} images...")

            # Pick a random image to augment
            img_path = np.random.choice(images)
            image = cv2.imread(str(img_path))
            
            if image is None or image.size == 0:
                print(f"Warning: Failed to load image {img_path}. Removing from list and skipping.")
                # Remove from list so we don't pick it again
                # Note: np.random.choice on a list of Paths returns a Path, but we need to find the index or value to remove
                # However, images is a list of Paths. np.random.choice returns a single element.
                # We need to be careful if images becomes empty.
                if img_path in images:
                    images.remove(img_path)
                if not images:
                    print(f"Error: No valid images left in class {cls} to augment.")
                    break
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply augmentation
            augmented = aug_pipeline(image=image)['image']
            
            # Save augmented image
            save_name = f"aug_{generated_count}_{img_path.name}"
            save_path = class_dir / save_name
            
            # Convert back to BGR for saving with OpenCV
            augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(save_path), augmented_bgr)
            
            generated_count += 1
            # pbar.update(1)
            
        # pbar.close()
        class_counts[cls] += generated_count

    # 3. Split into Train/Val/Test
    print("\nSplitting dataset into Train/Val/Test...")
    random.seed(42)
    
    classes = [d for d in TARGET_DIR.iterdir() if d.is_dir()]
    
    for cls_dir in classes:
        cls = cls_dir.name
        # Skip if it's already one of the split folders
        if cls in SPLIT.keys():
            continue
            
        imgs = list(cls_dir.glob("*"))
        # Filter for images
        imgs = [f for f in imgs if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        random.shuffle(imgs)
        
        n_total = len(imgs)
        n_train = int(SPLIT['train'] * n_total)
        n_val = int(SPLIT['val'] * n_total)
        
        splits = {
            'train': imgs[:n_train],
            'val': imgs[n_train:n_train+n_val],
            'test': imgs[n_train+n_val:]
        }
        
        print(f"Class: {cls}")
        for split_name, files in splits.items():
            split_dir = TARGET_DIR / split_name / cls
            split_dir.mkdir(parents=True, exist_ok=True)
            
            for f in files:
                shutil.move(str(f), str(split_dir / f.name))
                
            print(f"  {split_name}: {len(files)}")
            
        # Remove the original class directory
        try:
            cls_dir.rmdir()
        except OSError:
            print(f"Warning: Could not remove {cls_dir}, it might not be empty.")

if __name__ == "__main__":
    process_dataset()
