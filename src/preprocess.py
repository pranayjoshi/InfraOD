import os
from pre_process.data_processor import DataProcessor
from pre_process.data_splitter import DataSplitter
from pre_process.visualizer import Visualizer
from pre_process.llvip_processor import LLVIPProcessor
from pre_process.hituav_processor import HitUAVProcessor
from pre_process.config import LABEL_DIR

def process_camel_data():
    """Process Camel dataset"""
    print("\nProcessing Camel dataset...")
    processor = DataProcessor()
    all_labels = processor.process_sequences()
    processed_labels = processor.process_labels(all_labels)
    
    # Save processed labels
    processor.save_labels(processed_labels, os.path.join(LABEL_DIR, "ALL_LABELS.txt"))
    
    # Create visualizations and split data
    visualizer = Visualizer()
    visualizer.plot_class_distribution(processed_labels)
    
    splitter = DataSplitter(processed_labels)
    splitter.split_and_save(train_ratio=0.8, val_ratio=0.1)

def process_llvip_data():
    """Process LLVIP dataset"""
    print("\nProcessing LLVIP dataset...")
    processor = LLVIPProcessor()
    processor.process_data()

def process_hituav_data():
    """Process HitUAV dataset"""
    print("\nProcessing HitUAV dataset...")
    processor = HitUAVProcessor()
    processor.process_data()

def main():
    # Process Camel dataset
    process_camel_data()
    
    # Process LLVIP dataset
    process_llvip_data()
    
    # Process HitUAV dataset
    process_hituav_data()

if __name__ == "__main__":
    main()