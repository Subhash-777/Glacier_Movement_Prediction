"""
Main entry point for Glacier Movement Prediction
"""
import argparse
from pathlib import Path

from scripts.utils.config import Config
from scripts.preprocess.feature_extraction import extract_all_regions
from scripts.train.train import main as train_main
from scripts.evaluate.evaluate import main as evaluate_main
from scripts.evaluate.satellite_evaluation import main as satellite_eval_main

def main():
    parser = argparse.ArgumentParser(
        description='Glacier Movement Prediction'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['extract', 'train', 'evaluate', 'satellite_eval', 'all'],
        help='Execution mode'
    )
    
    parser.add_argument(
        '--model_type',
        type=str,
        default='timesformer',
        choices=['timesformer', 'simple_cnn'],
        help='Model architecture to use'
    )
    
    parser.add_argument(
        '--regions',
        type=str,
        nargs='+',
        default=None,
        help='Specific regions to process (default: all)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    args = parser.parse_args()
    
    # Update config based on arguments
    if args.model_type:
        Config.MODEL_TYPE = args.model_type
    
    if args.epochs:
        Config.NUM_EPOCHS = args.epochs
    
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    
    if args.learning_rate:
        Config.LEARNING_RATE = args.learning_rate
    
    # Print configuration
    print("\n" + "="*60)
    print("GLACIER MOVEMENT PREDICTION")
    print("="*60)
    print(f"\nMode: {args.mode}")
    print(f"Model: {Config.MODEL_TYPE}")
    print(f"Image Size: {Config.IMAGE_SIZE}")
    print(f"Num Frames: {Config.NUM_FRAMES}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Learning Rate: {Config.LEARNING_RATE}")
    print(f"Epochs: {Config.NUM_EPOCHS}")
    print("="*60 + "\n")
    
    # Execute based on mode
    if args.mode == 'extract':
        print("FEATURE EXTRACTION MODE\n")
        regions = args.regions if args.regions else Config.RGI_REGIONS
        extract_all_regions(regions)
    
    elif args.mode == 'train':
        print("TRAINING MODE\n")
        train_main()
    
    elif args.mode == 'evaluate':
        print("EVALUATION MODE\n")
        evaluate_main()
    
    elif args.mode == 'satellite_eval':
        print("SATELLITE EVALUATION MODE\n")
        satellite_eval_main()
    
    elif args.mode == 'all':
        print("FULL PIPELINE MODE\n")
        
        # Step 1: Feature extraction
        print("\n[1/4] Feature Extraction...")
        regions = args.regions if args.regions else Config.RGI_REGIONS
        extract_all_regions(regions)
        
        # Step 2: Training
        print("\n[2/4] Model Training...")
        train_main()
        
        # Step 3: Evaluation
        print("\n[3/4] Model Evaluation...")
        evaluate_main()
        
        # Step 4: Satellite evaluation
        print("\n[4/4] Satellite Evaluation...")
        satellite_eval_main()
        
        print("\n" + "="*60)
        print("FULL PIPELINE COMPLETED")
        print("="*60 + "\n")

if __name__ == "__main__":
    main()
