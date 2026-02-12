
import os, shutil, random
from pathlib import Path

SOURCE = Path("data/raw/PlantVillage")
DEST = Path("data/processed/PlantVillage")
SPLIT = {'train': 0.8, 'val': 0.1, 'test': 0.1}

random.seed(42)

for cls in os.listdir(SOURCE):
    cls_path = SOURCE / cls
    imgs = list(cls_path.glob("*.*"))  # Ensure only image files are considered
    random.shuffle(imgs)

    n_total = len(imgs)
    n_train = max(1, int(SPLIT['train'] * n_total))
    n_val = max(1, int(SPLIT['val'] * n_total))

    # Ensure no overlap and handle edge cases
    splits = {
        'train': imgs[:n_train],
        'val': imgs[n_train:n_train+n_val],
        'test': imgs[n_train+n_val:]
    }

    for split, files in splits.items():
        out_dir = DEST / split / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy(f, out_dir / f.name)

    # Log the distribution of images for each class
    print(f"Class: {cls}")
    print(f"  Train: {len(splits['train'])}")
    print(f"  Val: {len(splits['val'])}")
    print(f"  Test: {len(splits['test'])}")
