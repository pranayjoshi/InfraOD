import os
import shutil
import pandas as pd
import numpy as np
from pre_process.config import (
    CUR_DIR, DATASET_DIR, IMAGE_WIDTH, IMAGE_HEIGHT,
    ALL_IMAGES_DIR
)

class DataSplitter:
    def __init__(self, df):
        self.df = df
        self.image_ids = list(set(df["image_id"]))
        np.random.shuffle(self.image_ids)  # Shuffle image IDs
        
    def split_and_save(self, train_ratio=0.8, val_ratio=0.1):
        """Split data into train/val/test sets and save to respective directories"""
        n_samples = len(self.image_ids)
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        
        # Split image IDs
        train_ids = self.image_ids[:train_size]
        val_ids = self.image_ids[train_size:train_size + val_size]
        test_ids = self.image_ids[train_size + val_size:]
        
        # Process each split
        self._process_split(train_ids, "train")
        self._process_split(val_ids, "val")
        self._process_split(test_ids, "test")
        
        print(f"Split data into:")
        print(f"  Train: {len(train_ids)} images")
        print(f"  Val: {len(val_ids)} images")
        print(f"  Test: {len(test_ids)} images")
    
    def _create_directories(self, split_name):
        """Create necessary directories for the split"""
        for dir_type in ["labels", "images"]:
            path = os.path.join(DATASET_DIR, dir_type, split_name)
            if not os.path.exists(path):
                os.makedirs(path)
    
    def _process_split(self, image_ids, split_name):
        """Process a single split (train/val/test)"""
        # Create directories
        self._create_directories(split_name)
        
        for image_id in image_ids:
            # Get data for this image
            mini_df = self.df[self.df["image_id"] == image_id]
            
            # Save labels
            self._save_labels(mini_df, image_id, split_name)
            
            # Copy image
            self._copy_image(image_id, split_name)
    
    def _save_labels(self, mini_df, image_id, split_name):
        """Save normalized labels for an image"""
        output_path = os.path.join(DATASET_DIR, "labels", split_name, f"{image_id}.txt")
        
        with open(output_path, "w+") as fin:
            row = mini_df[["classes", "x_center", "y_center", "w", "h"]].astype(float).values
            
            # Normalize coordinates
            row[:, 1] /= IMAGE_WIDTH   # x_center
            row[:, 2] /= IMAGE_HEIGHT  # y_center
            row[:, 3] /= IMAGE_WIDTH   # width
            row[:, 4] /= IMAGE_HEIGHT  # height
            
            # Write to file
            for values in row.astype(str):
                fin.write(" ".join(values) + "\n")
    
    def _copy_image(self, image_id, split_name):
        """Copy image to the appropriate split directory"""
        source_dir = ALL_IMAGES_DIR
        target_dir = os.path.join(DATASET_DIR, "images", split_name)
        
        # Try jpg first, then png
        try:
            shutil.copy(
                os.path.join(source_dir, f"{image_id}.jpg"),
                os.path.join(target_dir, f"{image_id}.jpg")
            )
        except FileNotFoundError:
            shutil.copy(
                os.path.join(source_dir, f"{image_id}.png"),
                os.path.join(target_dir, f"{image_id}.png")
            ) 