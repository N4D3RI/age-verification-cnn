"""
Exploratory analysis of the ChaLearn Looking at People faces dataset.

Prints the dataset size, the age distribution, the image property audit, and
writes two figures. Everything the modelling decisions were based on.

    python explore_faces.py

Writes figures/age_distribution.png and figures/sample_faces.png, plus
results/eda_summary.json.
"""

import json
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

DATA_PATH = os.environ.get('FACES_PATH', '/datasets/faces/')
LABELS_PATH = DATA_PATH + 'labels.csv'
IMAGES_PATH = DATA_PATH + 'final_files/'

FIGURES_DIR = 'figures'
RESULTS_DIR = 'results'

AGE_BINS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 101]
SAMPLE_SIZE = 500
TARGET_AGES = [1, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100]
SEED = 12345


def load_labels():
    """Read labels.csv and report the basic shape of the dataset."""
    labels = pd.read_csv(LABELS_PATH)

    print('labels.csv shape:', labels.shape)
    print('files on disk:   ', len(os.listdir(IMAGES_PATH)))
    print()
    labels.info()
    print()
    print(labels.head())
    print()
    print('duplicated file_name rows:', labels['file_name'].duplicated().sum())

    return labels


def age_distribution(labels):
    """Describe the age column, plot it, and return the bucket shares."""
    print()
    print('=== age distribution ===')
    print(labels['real_age'].describe())
    print()
    print('min age:', labels['real_age'].min(), ' max age:', labels['real_age'].max())

    fig, axs = plt.subplots(1, 2, figsize=(14, 4))

    labels['real_age'].hist(bins=100, ax=axs[0])
    axs[0].set_title('Age distribution (bins=100)')
    axs[0].set_xlabel('real_age')
    axs[0].set_ylabel('number of photos')

    labels['real_age'].plot(kind='box', vert=False, ax=axs[1])
    axs[1].set_title('Age boxplot')
    axs[1].set_xlabel('real_age')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'age_distribution.png'), dpi=110)
    plt.close(fig)

    bucket = pd.cut(labels['real_age'], bins=AGE_BINS, right=False)
    share = (labels.groupby(bucket, observed=False)['real_age'].count() / len(labels) * 100).round(1)

    print()
    print('share of the dataset by age bucket, %')
    print(share)

    return share


def image_properties(labels):
    """Audit size and color mode on a random sample of files."""
    rng = np.random.RandomState(SEED)
    sample_names = rng.choice(labels['file_name'].values, size=SAMPLE_SIZE, replace=False)

    sizes = []
    modes = []
    for name in sample_names:
        with Image.open(IMAGES_PATH + name) as im:
            sizes.append(im.size)
            modes.append(im.mode)

    smallest = min(min(s) for s in sizes)
    largest = max(max(s) for s in sizes)

    print()
    print('=== image properties on a random sample of %d files ===' % SAMPLE_SIZE)
    print('most common sizes:')
    print(pd.Series(sizes).value_counts().head())
    print()
    print('smallest side seen:', smallest)
    print('largest side seen: ', largest)
    print('color modes:', pd.Series(modes).value_counts().to_dict())

    return {'smallest_side': int(smallest),
            'largest_side': int(largest),
            'color_modes': pd.Series(modes).value_counts().to_dict()}


def sample_faces(labels):
    """Save a grid of photos spread across the whole age range."""
    fig = plt.figure(figsize=(15, 9))

    for i, age in enumerate(TARGET_AGES):
        subset = labels[labels['real_age'] == age]
        if len(subset) == 0:
            continue
        row = subset.iloc[0]
        img = Image.open(IMAGES_PATH + row['file_name'])
        ax = fig.add_subplot(3, 5, i + 1)
        ax.imshow(img)
        ax.set_title('age ' + str(int(row['real_age'])) + '  ' + str(img.size))
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'sample_faces.png'), dpi=90)
    plt.close(fig)


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    labels = load_labels()
    share = age_distribution(labels)
    properties = image_properties(labels)
    sample_faces(labels)

    summary = {
        'photos': int(len(labels)),
        'missing_values': int(labels.isna().sum().sum()),
        'duplicate_filenames': int(labels['file_name'].duplicated().sum()),
        'age_min': int(labels['real_age'].min()),
        'age_max': int(labels['real_age'].max()),
        'age_mean': round(float(labels['real_age'].mean()), 2),
        'age_median': float(labels['real_age'].median()),
        'age_std': round(float(labels['real_age'].std()), 2),
        'age_bucket_share_pct': {str(k): float(v) for k, v in share.items()},
        'image_properties': properties,
    }

    with open(os.path.join(RESULTS_DIR, 'eda_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print()
    print('wrote figures/age_distribution.png')
    print('wrote figures/sample_faces.png')
    print('wrote results/eda_summary.json')


if __name__ == '__main__':
    main()
