import pandas as pd
import numpy as np
import os

"""
Improved Tomato Suitability Dataset Generator

This script generates a realistic dataset for training binary classification models
to determine if environmental conditions are suitable for tomato cultivation.

Key improvements over simple uniform generation:
1. Gaussian distributions centered on optimal ranges (more realistic)
2. Sensor noise added to all features
3. Overlapping boundary regions between classes (realistic ambiguity)
4. Label noise to simulate real-world annotation errors
5. Multiple negative sample categories (near-miss, random, adverse)
6. Feature correlations (e.g., temperature-humidity relationship)
"""

# Configuration
OUTPUT_PATH = 'data/processed/augmented_tomato_dataset.csv'
RANDOM_SEED = 42

# Dataset composition
NUM_OPTIMAL = 6000          # Clear positive cases
NUM_BORDERLINE_POS = 2000   # Positive but near boundaries
NUM_BORDERLINE_NEG = 2000   # Negative but near boundaries (challenging cases)
NUM_NEAR_MISS = 3000        # Close to optimal but clearly outside
NUM_ADVERSE = 3000          # Clearly bad conditions
NUM_RANDOM = 4000           # Random conditions

# Label noise rate (simulates annotation errors)
LABEL_NOISE_RATE = 0.02  # 2% of labels will be flipped

# Sensor noise standard deviation (as fraction of feature range)
SENSOR_NOISE_SCALE = 0.03  # 3% noise

# Optimal Ranges for Tomato (from agricultural research)
# N: 96 - 180 kg/ha
# P: 34 - 100 kg/ha
# K: 140 - 180 kg/ha
# Temp: 21-27C
# Humidity: 60%-85%
# pH: 6.0-7.0
# Rainfall: 600-1500 mm/year

RANGES = {
    'N': (96, 180),
    'P': (34, 100),
    'K': (140, 180),
    'temperature': (21, 27),
    'humidity': (60, 85),
    'ph': (6.0, 7.0),
    'rainfall': (600, 1500)
}

# Global bounds for random generation (realistic agricultural ranges)
GLOBAL_BOUNDS = {
    'N': (0, 250),
    'P': (0, 150),
    'K': (0, 250),
    'temperature': (5, 45),
    'humidity': (20, 100),
    'ph': (4.0, 9.0),
    'rainfall': (100, 3000)
}

# Adverse conditions for tomato (too extreme)
ADVERSE_RANGES = {
    'N': [(0, 40), (220, 250)],           # Too low or too high
    'P': [(0, 15), (120, 150)],
    'K': [(0, 80), (220, 250)],
    'temperature': [(5, 15), (35, 45)],   # Too cold or too hot
    'humidity': [(20, 40), (95, 100)],    # Too dry or too humid
    'ph': [(4.0, 5.0), (8.0, 9.0)],       # Too acidic or too alkaline
    'rainfall': [(100, 300), (2500, 3000)] # Too dry or flooding
}


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)


def add_sensor_noise(df, noise_scale=SENSOR_NOISE_SCALE):
    """Add realistic sensor noise to all features."""
    df_noisy = df.copy()
    for col in RANGES.keys():
        if col in df_noisy.columns:
            col_range = GLOBAL_BOUNDS[col][1] - GLOBAL_BOUNDS[col][0]
            noise = np.random.normal(0, col_range * noise_scale, len(df_noisy))
            df_noisy[col] = df_noisy[col] + noise
            # Clip to global bounds
            df_noisy[col] = df_noisy[col].clip(GLOBAL_BOUNDS[col][0], GLOBAL_BOUNDS[col][1])
    return df_noisy


def add_feature_correlations(df):
    """
    Add realistic correlations between features.
    - Higher temperature tends to correlate with lower humidity
    - Higher rainfall correlates with higher humidity
    """
    df_corr = df.copy()

    # Temperature-humidity inverse correlation
    temp_deviation = (df_corr['temperature'] - 24) / 10  # Normalized around optimal
    humidity_adjustment = -temp_deviation * 5  # Inverse relationship
    df_corr['humidity'] = (df_corr['humidity'] + humidity_adjustment).clip(
        GLOBAL_BOUNDS['humidity'][0], GLOBAL_BOUNDS['humidity'][1]
    )

    # Rainfall-humidity positive correlation
    rainfall_normalized = (df_corr['rainfall'] - 1000) / 1000
    humidity_rain_adj = rainfall_normalized * 3
    df_corr['humidity'] = (df_corr['humidity'] + humidity_rain_adj).clip(
        GLOBAL_BOUNDS['humidity'][0], GLOBAL_BOUNDS['humidity'][1]
    )

    return df_corr


def generate_optimal(n):
    """
    Generate clearly optimal conditions using Gaussian distribution
    centered on the middle of optimal ranges.
    """
    data = {}
    for col, (min_val, max_val) in RANGES.items():
        mean = (min_val + max_val) / 2
        std = (max_val - min_val) / 4  # 95% within range
        values = np.random.normal(mean, std, n)
        # Clip to stay within optimal range (with small buffer)
        buffer = (max_val - min_val) * 0.05
        data[col] = np.clip(values, min_val + buffer, max_val - buffer)

    df = pd.DataFrame(data)
    df['label'] = 1
    return df


def generate_borderline_positive(n):
    """
    Generate positive samples near the boundaries of optimal ranges.
    These are still suitable but marginal.
    """
    data = {col: [] for col in RANGES.keys()}

    for _ in range(n):
        # Randomly select 1-3 features to be near boundaries
        num_boundary_features = np.random.randint(1, 4)
        boundary_features = np.random.choice(list(RANGES.keys()), num_boundary_features, replace=False)

        for col, (opt_min, opt_max) in RANGES.items():
            if col in boundary_features:
                # Generate value near boundary (within 10% of edge)
                boundary_width = (opt_max - opt_min) * 0.15
                if np.random.random() < 0.5:
                    # Near lower boundary
                    val = np.random.uniform(opt_min, opt_min + boundary_width)
                else:
                    # Near upper boundary
                    val = np.random.uniform(opt_max - boundary_width, opt_max)
            else:
                # Normal optimal value
                mean = (opt_min + opt_max) / 2
                std = (opt_max - opt_min) / 4
                val = np.clip(np.random.normal(mean, std), opt_min, opt_max)

            data[col].append(val)

    df = pd.DataFrame(data)
    df['label'] = 1
    return df


def generate_borderline_negative(n):
    """
    Generate negative samples that are very close to optimal ranges.
    These are the challenging cases that make classification non-trivial.
    """
    data = {col: [] for col in RANGES.keys()}

    for _ in range(n):
        # Only 1-2 features slightly outside optimal
        num_outside = np.random.randint(1, 3)
        outside_features = np.random.choice(list(RANGES.keys()), num_outside, replace=False)

        for col, (opt_min, opt_max) in RANGES.items():
            range_width = opt_max - opt_min

            if col in outside_features:
                # Generate value just outside optimal range (within 15% outside)
                outside_margin = range_width * np.random.uniform(0.01, 0.15)
                if np.random.random() < 0.5:
                    # Just below minimum
                    val = opt_min - outside_margin
                else:
                    # Just above maximum
                    val = opt_max + outside_margin
                # Ensure within global bounds
                val = np.clip(val, GLOBAL_BOUNDS[col][0], GLOBAL_BOUNDS[col][1])
            else:
                # Inside optimal range (making it a near-miss)
                mean = (opt_min + opt_max) / 2
                std = (opt_max - opt_min) / 4
                val = np.clip(np.random.normal(mean, std), opt_min, opt_max)

            data[col].append(val)

    df = pd.DataFrame(data)
    df['label'] = 0
    return df


def generate_near_miss(n):
    """
    Generate data that is moderately outside optimal ranges.
    Multiple features are outside but not extremely so.
    """
    data = {col: [] for col in RANGES.keys()}

    for _ in range(n):
        # 2-4 features outside optimal
        num_outside = np.random.randint(2, 5)
        outside_features = np.random.choice(list(RANGES.keys()), num_outside, replace=False)

        for col, (opt_min, opt_max) in RANGES.items():
            range_width = opt_max - opt_min

            if col in outside_features:
                # 15-40% outside optimal range
                outside_margin = range_width * np.random.uniform(0.15, 0.40)
                if np.random.random() < 0.5:
                    val = opt_min - outside_margin
                else:
                    val = opt_max + outside_margin
                val = np.clip(val, GLOBAL_BOUNDS[col][0], GLOBAL_BOUNDS[col][1])
            else:
                # Could be inside or slightly outside
                if np.random.random() < 0.7:
                    # Inside optimal
                    val = np.random.uniform(opt_min, opt_max)
                else:
                    # Slightly outside
                    margin = range_width * np.random.uniform(0.05, 0.15)
                    if np.random.random() < 0.5:
                        val = opt_min - margin
                    else:
                        val = opt_max + margin
                    val = np.clip(val, GLOBAL_BOUNDS[col][0], GLOBAL_BOUNDS[col][1])

            data[col].append(val)

    df = pd.DataFrame(data)
    df['label'] = 0
    return df


def generate_adverse(n):
    """
    Generate clearly adverse conditions for tomato growth.
    """
    data = {}

    for col in RANGES.keys():
        values = []
        adverse_ranges = ADVERSE_RANGES[col]

        for _ in range(n):
            # Randomly pick one of the adverse ranges (low or high)
            selected_range = adverse_ranges[np.random.randint(0, len(adverse_ranges))]
            val = np.random.uniform(selected_range[0], selected_range[1])
            values.append(val)

        data[col] = values

    df = pd.DataFrame(data)
    df['label'] = 0
    return df


def generate_random_conditions(n):
    """
    Generate random conditions within global bounds.
    Some may accidentally fall in optimal ranges.
    """
    data = {}
    for col, (min_val, max_val) in GLOBAL_BOUNDS.items():
        data[col] = np.random.uniform(min_val, max_val, n)

    df = pd.DataFrame(data)

    # Determine labels based on how many features are in optimal range
    labels = []
    for idx, row in df.iterrows():
        features_in_range = 0
        for col, (opt_min, opt_max) in RANGES.items():
            if opt_min <= row[col] <= opt_max:
                features_in_range += 1

        # If most features (>=5 out of 7) are in optimal range, label as positive
        # This creates natural overlap and uncertainty
        if features_in_range >= 5:
            labels.append(1)
        else:
            labels.append(0)

    df['label'] = labels
    return df


def apply_label_noise(df, noise_rate=LABEL_NOISE_RATE):
    """
    Flip a small percentage of labels to simulate annotation errors.
    """
    df_noisy = df.copy()
    n_flip = int(len(df_noisy) * noise_rate)
    flip_indices = np.random.choice(df_noisy.index, n_flip, replace=False)
    df_noisy.loc[flip_indices, 'label'] = 1 - df_noisy.loc[flip_indices, 'label']
    return df_noisy


def main():
    set_seed(RANDOM_SEED)

    print("=" * 60)
    print("Tomato Suitability Dataset Generator (Improved)")
    print("=" * 60)

    print(f"\nGenerating {NUM_OPTIMAL} optimal records...")
    df_opt = generate_optimal(NUM_OPTIMAL)

    print(f"Generating {NUM_BORDERLINE_POS} borderline positive records...")
    df_border_pos = generate_borderline_positive(NUM_BORDERLINE_POS)

    print(f"Generating {NUM_BORDERLINE_NEG} borderline negative records...")
    df_border_neg = generate_borderline_negative(NUM_BORDERLINE_NEG)

    print(f"Generating {NUM_NEAR_MISS} near-miss records...")
    df_near = generate_near_miss(NUM_NEAR_MISS)

    print(f"Generating {NUM_ADVERSE} adverse condition records...")
    df_adverse = generate_adverse(NUM_ADVERSE)

    print(f"Generating {NUM_RANDOM} random condition records...")
    df_rand = generate_random_conditions(NUM_RANDOM)

    # Combine all datasets
    df_final = pd.concat([
        df_opt, df_border_pos, df_border_neg,
        df_near, df_adverse, df_rand
    ], ignore_index=True)

    print(f"\nAdding feature correlations...")
    df_final = add_feature_correlations(df_final)

    print(f"Adding sensor noise (scale={SENSOR_NOISE_SCALE})...")
    df_final = add_sensor_noise(df_final)

    print(f"Applying label noise (rate={LABEL_NOISE_RATE})...")
    df_final = apply_label_noise(df_final)

    # Shuffle
    df_final = df_final.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Ensure label is integer
    df_final['label'] = df_final['label'].astype(int)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df_final)} records to {OUTPUT_PATH}")

    # Statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)

    print("\nClass Distribution:")
    class_counts = df_final['label'].value_counts()
    print(class_counts)
    print(f"Positive ratio: {df_final['label'].mean():.2%}")

    print("\nFeature Statistics:")
    print(df_final.describe().round(2))

    # Check overlap
    print("\n" + "-" * 60)
    print("Boundary Analysis (features within optimal range):")
    print("-" * 60)

    for label in [0, 1]:
        subset = df_final[df_final['label'] == label]
        features_in_range = []
        for _, row in subset.iterrows():
            count = sum(1 for col, (opt_min, opt_max) in RANGES.items()
                       if opt_min <= row[col] <= opt_max)
            features_in_range.append(count)

        label_name = "Positive" if label == 1 else "Negative"
        print(f"\n{label_name} samples - Features in optimal range:")
        print(f"  Mean: {np.mean(features_in_range):.2f}")
        print(f"  Min:  {np.min(features_in_range)}")
        print(f"  Max:  {np.max(features_in_range)}")

    print("\nSample Data:")
    print(df_final.head(10))

    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()