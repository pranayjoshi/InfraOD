import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import os
from pre_process.config import ALL_IMAGES_DIR

class Visualizer:
    @staticmethod
    def plot_class_samples(df, class_id, num_samples=70, figsize=(50, 50)):
        """Plot sample images for a specific class"""
        plt.figure(figsize=figsize)
        images = df[df["classes"] == class_id]["image_id"].values
        
        for i in range(min(num_samples, len(images))):
            try:
                image = Image.open(os.path.join(ALL_IMAGES_DIR, f"{images[i]}.jpg"))
            except FileNotFoundError:
                try:
                    image = Image.open(os.path.join(ALL_IMAGES_DIR, f"{images[i]}.png"))
                except FileNotFoundError:
                    print(f"Warning: Image {images[i]} not found in {ALL_IMAGES_DIR}")
                    continue
            
            plt.subplot(7, 10, i+1)
            plt.imshow(image)
            plt.axis("off")
        
        plt.show()
    
    @staticmethod
    def plot_bounding_boxes(image_id, df):
        """Plot image with bounding boxes"""
        # Get image data
        image_path_jpg = os.path.join(ALL_IMAGES_DIR, f"{image_id}.jpg")
        image_path_png = os.path.join(ALL_IMAGES_DIR, f"{image_id}.png")
        
        if os.path.exists(image_path_jpg):
            image = Image.open(image_path_jpg)
        elif os.path.exists(image_path_png):
            image = Image.open(image_path_png)
        else:
            raise FileNotFoundError(f"Image {image_id} not found in {ALL_IMAGES_DIR}")
        
        # Convert image to numpy array
        x = np.array(image, dtype=np.uint8)
        
        # Create figure and axes
        fig, ax = plt.subplots(1)
        
        # Display the image
        ax.imshow(x)
        
        # Get bounding box data for this image
        image_data = df[df["image_id"] == image_id]
        if len(image_data) == 0:
            print(f"Warning: No bounding boxes found for image {image_id}")
            return
        
        # Draw each bounding box
        for _, row in image_data.iterrows():
            x_min = row["x_center"] - row["w"]/2
            y_min = row["y_center"] - row["h"]/2
            
            # Create Rectangle patch
            rect = patches.Rectangle(
                (x_min, y_min),
                row["w"],
                row["h"],
                linewidth=1,
                edgecolor='r',
                facecolor="none"
            )
            
            # Add the patch to the Axes
            ax.add_patch(rect)
            
            # Add class label
            plt.text(x_min, y_min-5, f"Class {row['classes']}", 
                    color='red', fontsize=8, 
                    bbox=dict(facecolor='white', alpha=0.7))
        
        plt.title(f"Image {image_id} with Bounding Boxes")
        plt.show()
    
    @staticmethod
    def plot_class_distribution(df):
        """Plot distribution of classes in the dataset"""
        class_counts = df.groupby("classes")["classes"].count()
        plt.figure(figsize=(10, 6))
        class_counts.plot(kind='bar')
        plt.title("Class Distribution in Dataset")
        plt.xlabel("Class ID")
        plt.ylabel("Count")
        
        # Add count labels on top of bars
        for i, count in enumerate(class_counts):
            plt.text(i, count, str(count), ha='center', va='bottom')
            
        plt.show() 