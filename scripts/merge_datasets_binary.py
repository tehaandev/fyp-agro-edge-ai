import os
import shutil
import random
from pathlib import Path

# Source paths
PLANTVILLAGE_SOURCE = Path("data/raw/PlantVillage")
CCMT_SOURCE = Path("data/raw/CCMT")
DEST = Path("data/processed/Merged_Binary")
SPLIT = {'train': 0.7, 'val': 0.15, 'test': 0.15}

random.seed(42)

# PlantVillage classes
PLANTVILLAGE_CATEGORIES = {
    'Healthy': [
        "Apple___healthy",
        "Blueberry___healthy", 
        "Cherry_(including_sour)___healthy",
        "Corn_(maize)___healthy",
        "Grape___healthy",
        "Peach___healthy",
        "Pepper__bell___healthy",
        "Pepper,_bell___healthy",
        "Potato___healthy",
        "Raspberry___healthy",
        "Soybean___healthy",
        "Strawberry___healthy",
        "Tomato___healthy",
        "Tomato_healthy"
    ],
    'Diseased': [
        # Apple
        "Apple___Apple_scab",
        "Apple___Black_rot", 
        "Apple___Cedar_apple_rust",
        # Cherry
        "Cherry_(including_sour)___Powdery_mildew",
        # Corn
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "Corn_(maize)___Common_rust_",
        "Corn_(maize)___Northern_Leaf_Blight",
        # Grape
        "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        # Citrus
        "Orange___Haunglongbing_(Citrus_greening)",
        # Peach
        "Peach___Bacterial_spot",
        # Pepper
        "Pepper__bell___Bacterial_spot",
        "Pepper,_bell___Bacterial_spot",
        # Potato
        "Potato___Early_blight",
        "Potato___Late_blight",
        # Squash
        "Squash___Powdery_mildew",
        # Strawberry
        "Strawberry___Leaf_scorch",
        # Tomato
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot",
        "Tomato___Tomato_mosaic_virus",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
    ]
}

# CCMT crops and their classes (simplified structure)
CCMT_CROPS = {
    'Cashew': ['anthracnose', 'gumosis', 'healthy', 'leaf miner', 'red rust'],
    'Cassava': ['bacterial blight', 'brown spot', 'green mite', 'healthy', 'mosaic'],
    'Maize': ['fall armyworm', 'grasshoper', 'healthy', 'leaf beetle', 'leaf blight', 'leaf spot', 'streak virus'],
    'Tomato': ['healthy', 'leaf blight', 'leaf curl', 'septoria leaf spot', 'verticulium wilt']
}

def classify_class(class_name):
    """Classify a class as healthy or diseased based on its name"""
    class_lower = class_name.lower()
    
    # Check for healthy indicators
    healthy_keywords = ['healthy']
    if any(keyword in class_lower for keyword in healthy_keywords):
        return 'Healthy'
    
    # Everything else is considered diseased
    return 'Diseased'

def collect_plantvillage_images():
    """Collect images from PlantVillage dataset"""
    healthy_images = []
    diseased_images = []
    
    print("Collecting PlantVillage images...")
    
    for category, classes in PLANTVILLAGE_CATEGORIES.items():
        for class_name in classes:
            class_path = PLANTVILLAGE_SOURCE / class_name
            
            if not class_path.exists():
                print(f"Warning: PlantVillage class directory {class_path} does not exist.")
                continue
            
            # Get all image files
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
            class_images = []
            
            for ext in image_extensions:
                class_images.extend(class_path.glob(ext))
                class_images.extend(class_path.glob(ext.upper()))
            
            if category == 'Healthy':
                healthy_images.extend(class_images)
            else:
                diseased_images.extend(class_images)
            
            print(f"  Found {len(class_images)} images in PlantVillage/{class_name} -> {category}")
    
    return healthy_images, diseased_images

def collect_ccmt_images():
    """Collect images from CCMT dataset"""
    healthy_images = []
    diseased_images = []
    
    print("\nCollecting CCMT images...")
    
    for crop, classes in CCMT_CROPS.items():
        crop_path = CCMT_SOURCE / crop
        
        if not crop_path.exists():
            print(f"Warning: CCMT crop directory {crop_path} does not exist.")
            continue
            
        # Process each class directly
        for class_name in classes:
            class_path = crop_path / class_name
            
            if not class_path.exists():
                print(f"Warning: CCMT class directory {class_path} does not exist.")
                continue
            
            # Get all image files
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
            
            print(f"  Found {len(class_images)} images in CCMT/{crop}/{class_name} -> {category}")
    
    return healthy_images, diseased_images

def create_unique_filename(img_file, source_type, extra_info=""):
    """Create a unique filename with source information"""
    if source_type == "plantvillage":
        # For PlantVillage: plantvillage_classname_filename
        class_name = img_file.parent.name
        unique_name = f"plantvillage_{class_name}_{img_file.name}"
    elif source_type == "ccmt":
        # For CCMT: ccmt_crop_class_filename
        # Extract crop and class from path
        parts = img_file.parts
        ccmt_idx = next(i for i, part in enumerate(parts) if part == "CCMT")
        if ccmt_idx + 2 < len(parts):
            crop = parts[ccmt_idx + 1]  # Skip just "CCMT"
            class_name = parts[ccmt_idx + 2]
            unique_name = f"ccmt_{crop}_{class_name}_{img_file.name}"
        else:
            unique_name = f"ccmt_{extra_info}_{img_file.name}"
    else:
        unique_name = img_file.name
    
    return unique_name

def split_and_copy_images(images_dict, category):
    """Split images into train/val/test and copy to destination"""
    all_images = []
    
    # Combine all images with their source info
    for source, images in images_dict.items():
        for img in images:
            all_images.append((img, source))
    
    if not all_images:
        print(f"Warning: No images found for category {category}")
        return None
    
    # Shuffle images
    random.shuffle(all_images)
    
    n_total = len(all_images)
    n_train = max(1, int(SPLIT['train'] * n_total))
    n_val = max(1, int(SPLIT['val'] * n_total))
    
    # Create splits
    splits = {
        'train': all_images[:n_train],
        'val': all_images[n_train:n_train+n_val],
        'test': all_images[n_train+n_val:]
    }
    
    # Copy files to destination
    for split, files in splits.items():
        out_dir = DEST / split / category
        out_dir.mkdir(parents=True, exist_ok=True)
        
        for img_file, source in files:
            try:
                # Create unique filename
                unique_name = create_unique_filename(img_file, source)
                dest_file = out_dir / unique_name
                
                # Handle potential duplicate names
                counter = 1
                original_name = unique_name
                while dest_file.exists():
                    stem = Path(original_name).stem
                    suffix = Path(original_name).suffix
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
    for source, images in images_dict.items():
        print(f"    From {source}: {len(images)} images")
    print(f"  Train: {len(splits['train'])}")
    print(f"  Val: {len(splits['val'])}")
    print(f"  Test: {len(splits['test'])}")
    
    return {split: len(files) for split, files in splits.items()}

def main():
    print("Starting merged dataset creation...")
    print(f"PlantVillage Source: {PLANTVILLAGE_SOURCE}")
    print(f"CCMT Source: {CCMT_SOURCE}")
    print(f"Destination: {DEST}")
    print(f"Split ratios: {SPLIT}")
    print("=" * 60)
    
    # Create destination directory
    DEST.mkdir(parents=True, exist_ok=True)
    
    # Collect images from both datasets
    pv_healthy, pv_diseased = collect_plantvillage_images()
    ccmt_healthy, ccmt_diseased = collect_ccmt_images()
    
    # Combine images from both sources
    all_healthy = {
        'plantvillage': pv_healthy,
        'ccmt': ccmt_healthy
    }
    
    all_diseased = {
        'plantvillage': pv_diseased,
        'ccmt': ccmt_diseased
    }
    
    print(f"\n" + "=" * 60)
    print("DATASET SUMMARY:")
    print(f"PlantVillage - Healthy: {len(pv_healthy)}, Diseased: {len(pv_diseased)}")
    print(f"CCMT - Healthy: {len(ccmt_healthy)}, Diseased: {len(ccmt_diseased)}")
    print(f"Total - Healthy: {len(pv_healthy) + len(ccmt_healthy)}, Diseased: {len(pv_diseased) + len(ccmt_diseased)}")
    
    # Check for balance
    total_healthy = len(pv_healthy) + len(ccmt_healthy)
    total_diseased = len(pv_diseased) + len(ccmt_diseased)
    
    # Balance the dataset by using equal numbers from both classes
    min_count = min(total_healthy, total_diseased)
    
    print(f"\nBalancing dataset...")
    print(f"Using {min_count} images from each class for balanced dataset")
    
    # Sample equal numbers from each class
    all_healthy_list = pv_healthy + ccmt_healthy
    all_diseased_list = pv_diseased + ccmt_diseased
    
    random.shuffle(all_healthy_list)
    random.shuffle(all_diseased_list)
    
    balanced_healthy_pv = []
    balanced_healthy_ccmt = []
    balanced_diseased_pv = []
    balanced_diseased_ccmt = []
    
    # Distribute balanced samples back to source categories
    healthy_count = 0
    for img in all_healthy_list[:min_count]:
        if 'PlantVillage' in str(img):
            balanced_healthy_pv.append(img)
        else:
            balanced_healthy_ccmt.append(img)
        healthy_count += 1
    
    diseased_count = 0
    for img in all_diseased_list[:min_count]:
        if 'PlantVillage' in str(img):
            balanced_diseased_pv.append(img)
        else:
            balanced_diseased_ccmt.append(img)
        diseased_count += 1
    
    balanced_healthy = {
        'plantvillage': balanced_healthy_pv,
        'ccmt': balanced_healthy_ccmt
    }
    
    balanced_diseased = {
        'plantvillage': balanced_diseased_pv,
        'ccmt': balanced_diseased_ccmt
    }
    
    print(f"\nAfter balancing:")
    print(f"Healthy - PlantVillage: {len(balanced_healthy_pv)}, CCMT: {len(balanced_healthy_ccmt)}")
    print(f"Diseased - PlantVillage: {len(balanced_diseased_pv)}, CCMT: {len(balanced_diseased_ccmt)}")
    print("=" * 60)
    
    # Split and copy images
    healthy_splits = split_and_copy_images(balanced_healthy, 'Healthy')
    diseased_splits = split_and_copy_images(balanced_diseased, 'Diseased')
    
    print("\n" + "=" * 60)
    print("MERGED BINARY DATASET CREATION COMPLETE!")
    print("=" * 60)
    
    # Final summary
    if healthy_splits and diseased_splits:
        total_train = healthy_splits['train'] + diseased_splits['train']
        total_val = healthy_splits['val'] + diseased_splits['val']
        total_test = healthy_splits['test'] + diseased_splits['test']
        total_all = total_train + total_val + total_test
        
        print(f"\nFinal Distribution:")
        print(f"  Train: {total_train} images ({total_train/total_all*100:.1f}%)")
        print(f"    - Healthy: {healthy_splits['train']}")
        print(f"    - Diseased: {diseased_splits['train']}")
        print(f"  Val: {total_val} images ({total_val/total_all*100:.1f}%)")
        print(f"    - Healthy: {healthy_splits['val']}")
        print(f"    - Diseased: {diseased_splits['val']}")
        print(f"  Test: {total_test} images ({total_test/total_all*100:.1f}%)")
        print(f"    - Healthy: {healthy_splits['test']}")
        print(f"    - Diseased: {diseased_splits['test']}")
        print(f"  Total: {total_all} images")
        
        print(f"\nDataset saved to: {DEST}")
        print("Ready for training!")

if __name__ == "__main__":
    main()
