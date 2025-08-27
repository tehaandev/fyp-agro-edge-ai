import os, shutil, random
from pathlib import Path

SOURCE = Path("data/raw/PlantVillage")
DEST = Path("data/processed/PlantVillage_Binary")
SPLIT = {'train': 0.6, 'val': 0.2, 'test': 0.2}

random.seed(42)

# Define binary classification categories
CATEGORIES = {
    'Healthy': [
        # Common healthy classes across PlantVillage
        "Apple___healthy",
        "Blueberry___healthy",
        "Cherry_(including_sour)___healthy",
        "Corn_(maize)___healthy",
        "Grape___healthy",
        "Peach___healthy",
        "Pepper__bell___healthy",
        "Pepper,_bell___healthy",  # alt naming in some dumps
        "Potato___healthy",
        "Raspberry___healthy",
        "Soybean___healthy",
        "Strawberry___healthy",
        "Tomato___healthy",
        "Tomato_healthy"  # alt naming found in some folders
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
        # Pepper (two naming variants)
        "Pepper__bell___Bacterial_spot",
        "Pepper,_bell___Bacterial_spot",
        # Potato
        "Potato___Early_blight",
        "Potato___Late_blight",
        # Squash
        "Squash___Powdery_mildew",
        # Strawberry
        "Strawberry___Leaf_scorch",
        # Tomato (canonical PlantVillage names)
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot",
        "Tomato___Tomato_mosaic_virus",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        # Tomato (alt naming variants sometimes present)
        "Tomato__Target_Spot",
        "Tomato__Tomato_mosaic_virus",
        "Tomato__Tomato_YellowLeaf__Curl_Virus",
        "Tomato_Bacterial_spot",
        "Tomato_Early_blight",
        "Tomato_Late_blight",
        "Tomato_Leaf_Mold",
        "Tomato_Septoria_leaf_spot",
        "Tomato_Spider_mites_Two_spotted_spider_mite"
    ]
}

for category, classes in CATEGORIES.items():
    all_images = []
    for cls in classes:
        cls_path = SOURCE / cls
        if not cls_path.exists():
            print(f"Warning: Class directory {cls_path} does not exist.")
            continue
        all_images.extend(cls_path.glob("*.*"))  # Collect all images from the class

    random.shuffle(all_images)

    n_total = len(all_images)
    n_train = max(1, int(SPLIT['train'] * n_total))
    n_val = max(1, int(SPLIT['val'] * n_total))

    # Ensure no overlap and handle edge cases
    splits = {
        'train': all_images[:n_train],
        'val': all_images[n_train:n_train+n_val],
        'test': all_images[n_train+n_val:]
    }

    for split, files in splits.items():
        out_dir = DEST / split / category
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy(f, out_dir / f.name)

    # Log the distribution of images for each category
    print(f"Category: {category}")
    print(f"  Train: {len(splits['train'])}")
    print(f"  Val: {len(splits['val'])}")
    print(f"  Test: {len(splits['test'])}")
