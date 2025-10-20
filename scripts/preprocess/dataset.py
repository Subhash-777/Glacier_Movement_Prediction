"""
PyTorch Dataset for Glacier Movement Prediction
Loads preprocessed features and generates training samples
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
from pathlib import Path
from scripts.utils.config import Config
from scripts.utils.data_utils import GlacierDataUtils
from scripts.preprocess.target_generation import TargetGenerator
import albumentations as A

class GlacierDataset(Dataset):
    
    def __init__(self, regions, num_frames=6, image_size=128, 
                 mode='train', target_mode='balanced', augment=True):
        """
        Args:
            regions: List of RGI region names
            num_frames: Number of temporal frames
            image_size: Target image size
            mode: 'train', 'val', or 'test'
            target_mode: 'strict', 'balanced', or 'relaxed'
            augment: Whether to apply data augmentation
        """
        self.regions = regions
        self.num_frames = num_frames
        self.image_size = image_size
        self.mode = mode
        self.augment = augment and (mode == 'train')
        
        self.utils = GlacierDataUtils()
        self.target_generator = TargetGenerator(mode=target_mode)
        
        # Load all features
        self.samples = self._load_samples()
        
        # Augmentation pipeline
        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.2),
            ])
        else:
            self.transform = None
    
    def _load_samples(self):
        """Load and prepare all training samples"""
        samples = []
        
        for region in self.regions:
            # Load static features
            static_path = Config.STATIC_FEATURES_DIR / f"{region}_static.pkl"
            if not static_path.exists():
                print(f"Warning: Static features not found for {region}")
                continue
                
            try:
                with open(static_path, 'rb') as f:
                    static_features = pickle.load(f)
            except Exception as e:
                print(f"Error loading static features for {region}: {e}")
                continue
            
            # Load dynamic features
            dynamic_path = Config.DYNAMIC_FEATURES_DIR / f"{region}_dynamic.pkl"
            if not dynamic_path.exists():
                print(f"Warning: Dynamic features not found for {region}")
                continue
                
            try:
                with open(dynamic_path, 'rb') as f:
                    dynamic_features = pickle.load(f)
            except Exception as e:
                print(f"Error loading dynamic features for {region}: {e}")
                continue
            
            if len(dynamic_features) < self.num_frames:
                print(f"Warning: Insufficient frames for {region} ({len(dynamic_features)} < {self.num_frames})")
                # Pad with duplicates if we have at least 1 frame
                if len(dynamic_features) > 0:
                    while len(dynamic_features) < self.num_frames:
                        dynamic_features.append(dynamic_features[-1])
                else:
                    continue
            
            # Create temporal sequences
            num_sequences = max(1, len(dynamic_features) - self.num_frames + 1)
            
            for i in range(num_sequences):
                # Get sequence
                end_idx = min(i + self.num_frames, len(dynamic_features))
                sequence = dynamic_features[i:end_idx]
                
                # Pad if needed
                while len(sequence) < self.num_frames:
                    sequence.append(sequence[-1])
                
                sample = {
                    'region': region,
                    'static': static_features,
                    'dynamic_sequence': sequence,
                    'index': i
                }
                
                samples.append(sample)
        
        print(f"Loaded {len(samples)} samples from {len(self.regions)} regions")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a training sample"""
        try:
            sample = self.samples[idx]
            
            # Extract static features
            static = sample['static']
            dem = static.get('dem', None)
            slope = static.get('slope', None)
            aspect = static.get('aspect', None)
            depressions = static.get('depressions', None)
            
            # Create defaults if missing
            if dem is None:
                dem = np.zeros((512, 512), dtype=np.float32)
            if slope is None:
                slope = np.zeros_like(dem, dtype=np.float32)
            if aspect is None:
                aspect = np.zeros_like(dem, dtype=np.float32)
            if depressions is None:
                depressions = np.zeros_like(dem, dtype=np.float32)
            
            # Extract dynamic features (temporal sequence)
            dynamic_sequence = sample['dynamic_sequence']
            
            velocity_frames = []
            vx_frames = []
            vy_frames = []
            divergence_frames = []
            
            for frame in dynamic_sequence:
                v_mag = frame.get('v_magnitude', None)
                vx = frame.get('vx', None)
                vy = frame.get('vy', None)
                div = frame.get('divergence', None)
                
                # Use zeros if data is missing
                if v_mag is None:
                    v_mag = np.zeros((512, 512), dtype=np.float32)
                if vx is None:
                    vx = np.zeros_like(v_mag, dtype=np.float32)
                if vy is None:
                    vy = np.zeros_like(v_mag, dtype=np.float32)
                if div is None:
                    div = np.zeros_like(v_mag, dtype=np.float32)
                
                velocity_frames.append(v_mag)
                vx_frames.append(vx)
                vy_frames.append(vy)
                divergence_frames.append(div)
            
            # Stack temporal dimension
            velocity_sequence = np.stack(velocity_frames, axis=0)  # (T, H, W)
            vx_sequence = np.stack(vx_frames, axis=0)
            vy_sequence = np.stack(vy_frames, axis=0)
            divergence_sequence = np.stack(divergence_frames, axis=0)
            
            # ===== RESIZE EVERYTHING TO TARGET SIZE FIRST =====
            dem = self.utils.resize_array(dem, (self.image_size, self.image_size))
            slope = self.utils.resize_array(slope, (self.image_size, self.image_size))
            aspect = self.utils.resize_array(aspect, (self.image_size, self.image_size))
            depressions = self.utils.resize_array(depressions, (self.image_size, self.image_size))
            
            velocity_sequence = self.utils.resize_array(velocity_sequence, (self.image_size, self.image_size))
            vx_sequence = self.utils.resize_array(vx_sequence, (self.image_size, self.image_size))
            vy_sequence = self.utils.resize_array(vy_sequence, (self.image_size, self.image_size))
            divergence_sequence = self.utils.resize_array(divergence_sequence, (self.image_size, self.image_size))
            
            # Normalize AFTER resizing
            dem = self.utils.normalize_data(dem, method='minmax')
            slope = self.utils.normalize_data(slope, method='minmax')
            aspect = self.utils.normalize_data(aspect, method='minmax')
            velocity_sequence = self.utils.normalize_data(velocity_sequence, method='log')
            vx_sequence = self.utils.normalize_data(vx_sequence, method='zscore')
            vy_sequence = self.utils.normalize_data(vy_sequence, method='zscore')
            divergence_sequence = self.utils.normalize_data(divergence_sequence, method='zscore')
            
            # Generate target using RESIZED data (all same size now)
            # Use the last frame of velocity for target generation
            last_velocity = velocity_sequence[-1]  # Already 128x128
            
            # For target generation, we need the DEM in its original scale
            # So we denormalize and scale it appropriately
            dem_for_target = dem * 5000  # Approximate scaling for DEM (adjust based on your data)
            
            target = self.target_generator.generate_lake_targets(
                last_velocity,
                dem_for_target,
                depressions
            )
            
            # Ensure target is the right size
            if target.shape != (self.image_size, self.image_size):
                target = self.utils.resize_array(target, (self.image_size, self.image_size))
            
            # Apply augmentation
            if self.augment and self.transform:
                # Prepare data for albumentations
                augmented = self.transform(
                    image=velocity_sequence[-1],
                    mask=target
                )
                target = augmented['mask']
            
            # Prepare model input
            # Stack features: [T, C, H, W] where C includes velocity, vx, vy, divergence
            velocity_input = np.stack([
                velocity_sequence,
                vx_sequence,
                vy_sequence,
                divergence_sequence
            ], axis=1)  # (T, 4, H, W)
            
            # Static features as additional channels (replicated across time)
            static_input = np.stack([dem, slope, aspect, depressions], axis=0)  # (4, H, W)
            static_input = np.repeat(static_input[np.newaxis, :, :, :], self.num_frames, axis=0)  # (T, 4, H, W)
            
            # Concatenate: (T, 8, H, W)
            model_input = np.concatenate([velocity_input, static_input], axis=1)
            
            # Convert to tensors
            model_input = torch.from_numpy(model_input).float()
            target = torch.from_numpy(target).float().unsqueeze(0)  # (1, H, W)
            
            return {
                'input': model_input,
                'target': target,
                'region': sample['region'],
                'index': sample['index']
            }
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            
            # Return a dummy sample instead of None
            dummy_input = torch.zeros(self.num_frames, 8, self.image_size, self.image_size)
            dummy_target = torch.zeros(1, self.image_size, self.image_size)
            return {
                'input': dummy_input,
                'target': dummy_target,
                'region': 'error',
                'index': -1
            }


def create_dataloaders(train_regions, val_regions, config=Config):
    """Create train and validation dataloaders"""
    
    train_dataset = GlacierDataset(
        regions=train_regions,
        num_frames=config.NUM_FRAMES,
        image_size=config.IMAGE_SIZE,
        mode='train',
        target_mode=config.TARGET_MODE,
        augment=config.AUGMENTATION
    )
    
    val_dataset = GlacierDataset(
        regions=val_regions,
        num_frames=config.NUM_FRAMES,
        image_size=config.IMAGE_SIZE,
        mode='val',
        target_mode=config.TARGET_MODE,
        augment=False
    )
    
    # Use num_workers=0 to avoid multiprocessing issues on Windows
    num_workers = 0 if config.NUM_WORKERS > 0 else 0
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY if torch.cuda.is_available() else False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader
