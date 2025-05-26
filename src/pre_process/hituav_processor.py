import os
import shutil
import yaml
import numpy as np
from pre_process.config import (
    CUR_DIR, DATASET_DIR, IMAGE_WIDTH, IMAGE_HEIGHT
)

class HitUAVProcessor:
    def __init__(self):
        self.hituav_dir = os.path.join(CUR_DIR, "HitUAV_data")
        self.images_dir = os.path.join(self.hituav_dir, "images")
        self.labels_dir = os.path.join(self.hituav_dir, "labels")
        
        # Load dataset config
        with open(os.path.join(self.hituav_dir, "dataset.yaml"), 'r') as f:
            self.config = yaml.safe_load(f)
        
    def process_data(self):
        """Process HitUAV data and create train/val/test splits"""
        # Create dataset directory structure
        self._create_dataset_dirs()
        
        # Get all image files
        all_images = []
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(self.images_dir, split)
            if os.path.exists(split_dir):
                images = [f for f in os.listdir(split_dir) if f.endswith(('.jpg', '.png'))]
                all_images.extend([(split, img) for img in images])
        
        # Shuffle all images
        np.random.shuffle(all_images)
        
        # Calculate split sizes
        n_samples = len(all_images)
        train_size = int(0.8 * n_samples)
        val_size = int(0.1 * n_samples)
        
        # Split data
        train_images = all_images[:train_size]
        val_images = all_images[train_size:train_size + val_size]
        test_images = all_images[train_size + val_size:]
        
        # Process each split
        self._process_split(train_images, "train")
        self._process_split(val_images, "val")
        self._process_split(test_images, "test")
        
        print(f"Processed HitUAV dataset with {self.config['nc']} classes:")
        for class_id, class_name in self.config['names'].items():
            print(f"  Class {class_id}: {class_name}")
        print(f"\nSplit data into:")
        print(f"  Train: {len(train_images)} images")
        print(f"  Val: {len(val_images)} images")
        print(f"  Test: {len(test_images)} images")
    
    def _create_dataset_dirs(self):
        """Create necessary directories for the dataset"""
        for split in ["train", "val", "test"]:
            for dir_type in ["images", "labels"]:
                path = os.path.join(DATASET_DIR, dir_type, split)
                if not os.path.exists(path):
                    os.makedirs(path)
    
    def _process_split(self, image_list, split_name):
        """Process a single split (train/val/test)"""
        dst_img_dir = os.path.join(DATASET_DIR, "images", split_name)
        dst_label_dir = os.path.join(DATASET_DIR, "labels", split_name)
        
        for src_split, img_file in image_list:
            # Get corresponding label file
            label_file = os.path.splitext(img_file)[0] + '.txt'
            
            # Copy image
            src_img = os.path.join(self.images_dir, src_split, img_file)
            dst_img = os.path.join(dst_img_dir, img_file)
            shutil.copy2(src_img, dst_img)
            
            # Process and copy label
            src_label = os.path.join(self.labels_dir, src_split, label_file)
            dst_label = os.path.join(dst_label_dir, label_file)
            
            if os.path.exists(src_label):
                # Read and process label
                with open(src_label, 'r') as f:
                    lines = f.readlines()
                
                # Write processed label
                with open(dst_label, 'w') as f:
                    for line in lines:
                        # HitUAV labels are already in YOLO format
                        f.write(line) 