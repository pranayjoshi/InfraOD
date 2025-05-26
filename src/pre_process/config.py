import os

# Directory paths
CUR_DIR = os.pardir
IMAGE_DIR = os.path.join(CUR_DIR, "Camel_data", "images")
LABEL_DIR = os.path.join(CUR_DIR, "Camel_data", "labels")
ALL_IMAGES_DIR = os.path.join(CUR_DIR, "Camel_data", "all_images")
DATASET_DIR = os.path.join(CUR_DIR, "datasets", "infrared")

# Image extensions for each sequence
EXTENSIONS = {
    1: ".jpg", 2: ".jpg", 3: ".jpg", 4: ".jpg", 5: ".jpg",
    6: ".jpg", 7: ".jpg", 8: ".jpg", 9: ".png", 10: ".png",
    11: ".jpg", 12: ".jpg", 13: ".jpg", 14: ".jpg", 15: ".jpg",
    16: ".jpg", 17: ".jpg", 18: ".jpg", 19: ".jpg", 20: ".jpg",
    21: ".jpg"
}

# Class mapping
CLASS_MAP = {
    1: "Person",
    2: "Bicycle",
    3: "Car"
}

# Image dimensions
IMAGE_WIDTH = 336
IMAGE_HEIGHT = 256

# Column names for DataFrame
COLUMN_NAMES = ["image_id", "x", "y", "w", "h", "x_center", "y_center", "classes"] 