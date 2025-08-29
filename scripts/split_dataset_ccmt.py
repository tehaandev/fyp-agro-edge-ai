import os
import shutil
import random
from pathlib import Path

SOURCE = Path("data/raw/CCMT/augmented")
DEST = Path("data/processed/CCMT_Binary")
SPLIT = {'train': 0.7, 'val': 0.15, 'test': 0.15}

random.seed(42)

# Define crops and their classes
CROPS = {
    'Cashew': {
        'train_set': ['anthracnose3102', 'gumosis1714', 'healthy5877', 'leaf miner3466', 'red rust4751'],
        'test_set': ['anthracnose', 'gumosis', 'healthy', 'leaf miner', 'red rust']
    },
    'Cassava': {
        'train_set': ['bacterial blight', 'bacterial blight3241', 'brown spot', 'green mite', 'healthy', 'mosaic'],
        'test_set': ['bacterial blight', 'brown spot', 'green mite', 'healthy', 'mosaic']
    },
    'Maize': {
        'train_set': ['fall armyworm', 'grasshoper', 'healthy', 'leaf beetle', 'leaf blight', 'leaf spot', 'streak virus'],
        'test_set': ['fall armyworm', 'grasshoper', 'healthy', 'leaf beetle', 'leaf blight', 'leaf spot', 'streak virus']
    },
    'Tomato': {
        'train_set': ['healthy', 'leaf blight', 'leaf curl', 'septoria leaf spot', 'verticulium wilt'],
        'test_set': ['healthy', 'leaf blight', 'leaf curl', 'septoria leaf spot', 'verticulium wilt']
    }
}

# Define binary classification categories
def classify_class(class_name):
    """Classify a class as healthy or diseased based on its name"""
    class_lower = class_name.lower()
    
    # Check for healthy indicators
    healthy_keywords = ['healthy']
    if any(keyword in class_lower for keyword in healthy_keywords):
        return 'Healthy'
    
    # Everything else is considered diseased
    return 'Diseased'

def collect_all_images():
    """Collect all images from all crops and organize by binary category"""
    healthy_images = []
    diseased_images = []
    
    for crop, sets in CROPS.items():
        crop_path = SOURCE / crop
        
        if not crop_path.exists():
            print(f"Warning: Crop directory {crop_path} does not exist.")
            continue
            
        # Process both train_set and test_set
        for set_name, classes in sets.items():
            set_path = crop_path / set_name
            
            if not set_path.exists():
                print(f"Warning: Set directory {set_path} does not exist.")
                continue
                
            for class_name in classes:
                class_path = set_path / class_name
                
                if not class_path.exists():
                    print(f"Warning: Class directory {class_path} does not exist.")
                    continue
                
                # Get all image files (common formats)
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
                class_images = []
                
                for ext in image_extensions:
                    class_images.extend(class_path.glob(ext))
                    class_images.extend(class_path.glob(ext.upper()))
                
                # Classify and add to appropriate list
                category = classify_class(class_name)
                if category == 'Healthy':
                    healthy_images.extend(class_images)
                else:
                    diseased_images.extend(class_images)
                
                print(f"Found {len(class_images)} images in {crop}/{set_name}/{class_name} -> {category}")
    
    return healthy_images, diseased_images

def split_and_copy_images(images, category):
    """Split images into train/val/test and copy to destination"""
    if not images:
        print(f"Warning: No images found for category {category}")
        return
    
    # Shuffle images
    random.shuffle(images)
    
    n_total = len(images)
    n_train = max(1, int(SPLIT['train'] * n_total))
    n_val = max(1, int(SPLIT['val'] * n_total))
    
    # Create splits
    splits = {
        'train': images[:n_train],
        'val': images[n_train:n_train+n_val],
        'test': images[n_train+n_val:]
    }
    
    # Copy files to destination
    for split, files in splits.items():
        out_dir = DEST / split / category
        out_dir.mkdir(parents=True, exist_ok=True)
        
        for img_file in files:
            try:
                # Create unique name with path info
                rel_path = img_file.relative_to(SOURCE)
                parts = rel_path.parts
                if len(parts) >= 3:
                    crop, set_name, class_name = parts[0], parts[1], parts[2]
                    unique_name = f"{crop}_{set_name}_{class_name}_{img_file.name}"
                else:
                    unique_name = img_file.name
                
                dest_file = out_dir / unique_name
                
                # Handle potential duplicate names
                counter = 1
                while dest_file.exists():
                    stem = Path(unique_name).stem
                    suffix = Path(unique_name).suffix
                    unique_name = f"{stem}_{counter}{suffix}"
                    dest_file = out_dir / unique_name
                    counter += 1
                
                shutil.copy2(img_file, dest_file)
                
            except Exception as e:
                print(f"Error copying {img_file}: {e}")
                continue
    
    # Log the distribution
    print(f"\nCategory: {category}")
    print(f"  Total images: {n_total}")
    print(f"  Train: {len(splits['train'])}")
    print(f"  Val: {len(splits['val'])}")
    print(f"  Test: {len(splits['test'])}")
    
    return splits

def main():
    print("Starting CCMT dataset binary classification split...")
    print(f"Source: {SOURCE}")
    print(f"Destination: {DEST}")
    print(f"Split ratios: {SPLIT}")
    print("-" * 50)
    
    # Create destination directory
    DEST.mkdir(parents=True, exist_ok=True)
    
    # Collect all images
    healthy_images, diseased_images = collect_all_images()
    
    print(f"\nOriginal distribution:")
    print(f"Total healthy images found: {len(healthy_images)}")
    print(f"Total diseased images found: {len(diseased_images)}")
    
    # Balance the dataset by using equal numbers from both classes
    min_count = min(len(healthy_images), len(diseased_images))
    
    print(f"\nBalancing dataset...")
    print(f"Using {min_count} images from each class for balanced dataset")
    
    # Randomly sample equal numbers from each class
    random.shuffle(healthy_images)
    random.shuffle(diseased_images)
    
    balanced_healthy = healthy_images[:min_count]
    balanced_diseased = diseased_images[:min_count]
    
    print(f"\nAfter balancing:")
    print(f"Healthy images: {len(balanced_healthy)}")
    print(f"Diseased images: {len(balanced_diseased)}")
    print("-" * 50)
    
    # Split and copy images
    healthy_splits = split_and_copy_images(balanced_healthy, 'Healthy')
    diseased_splits = split_and_copy_images(balanced_diseased, 'Diseased')
    
    print("\n" + "=" * 50)
    print("CCMT Binary Dataset Split Complete!")
    print("=" * 50)
    
    # Final summary
    if healthy_splits and diseased_splits:
        total_train = len(healthy_splits['train']) + len(diseased_splits['train'])
        total_val = len(healthy_splits['val']) + len(diseased_splits['val'])
        total_test = len(healthy_splits['test']) + len(diseased_splits['test'])
        total_all = total_train + total_val + total_test
        
        print(f"\nFinal Distribution:")
        print(f"  Train: {total_train} images ({total_train/total_all*100:.1f}%)")
        print(f"    - Healthy: {len(healthy_splits['train'])}")
        print(f"    - Diseased: {len(diseased_splits['train'])}")
        print(f"  Val: {total_val} images ({total_val/total_all*100:.1f}%)")
        print(f"    - Healthy: {len(healthy_splits['val'])}")
        print(f"    - Diseased: {len(diseased_splits['val'])}")
        print(f"  Test: {total_test} images ({total_test/total_all*100:.1f}%)")
        print(f"    - Healthy: {len(healthy_splits['test'])}")
        print(f"    - Diseased: {len(diseased_splits['test'])}")
        print(f"  Total: {total_all} images")
        
    # Clean up temporary augmented directory (if it exists from previous runs)
    temp_aug_dir = Path("temp_augmented_healthy")
    if temp_aug_dir.exists():
        print(f"\nNote: Temporary augmented directory {temp_aug_dir} still exists from previous runs.")
        print("You can delete it manually if no longer needed.")

if __name__ == "__main__":
    main()
