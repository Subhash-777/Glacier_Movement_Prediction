"""
Configuration file for Glacier Movement Prediction
Bharatiya Antariksh Hackathon 2025
"""
import os
from pathlib import Path

class Config:
    # Base paths
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    FEATURES_DIR = BASE_DIR / "features"
    MODELS_DIR = BASE_DIR / "models"
    EVAL_DIR = BASE_DIR / "evaluation"
    
    # Data paths
    VELOCITY_DIR = DATA_DIR / "velocity" / "ITS_LIVE"
    OUTLINES_DIR = DATA_DIR / "outlines"
    CENTERLINES_DIR = DATA_DIR / "centerlines"
    DEM_DIR = DATA_DIR / "dem"
    SATELLITE_DIR = DATA_DIR / "satellite"
    MASS_BALANCE_DIR = DATA_DIR / "mass_balance"
    
    # Satellite data paths
    SENTINEL1_DIR = SATELLITE_DIR / "Sentinel1"
    SENTINEL2_DIR = SATELLITE_DIR / "Sentinel2"
    LANDSAT_DIR = SATELLITE_DIR / "Landsat"
    
    # Feature directories
    STATIC_FEATURES_DIR = FEATURES_DIR / "static"
    DYNAMIC_FEATURES_DIR = FEATURES_DIR / "dynamic"
    
    # Model paths
    CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
    ARCHIVE_DIR = MODELS_DIR / "archives"
    
    # Evaluation paths
    METRICS_DIR = EVAL_DIR / "metrics"
    PLOTS_DIR = EVAL_DIR / "plots"
    
    # RGI Regions - 13 main regions for training
    RGI_REGIONS = [
        'RGI01_Alaska',
        'RGI02_WesternCanadaUSA',
        'RGI03_ArcticCanadaNorth',
        'RGI04_ArcticCanadaSouth',
        'RGI05_GreenlandPeriphery',
        'RGI06_Iceland',
        'RGI07_SvalbardJanMayen',
        'RGI08_Scandinavia',
        'RGI09_RussianArctic',
        'RGI14_Karakoram',
        'RGI17_SouthernAndes',
        'RGI18_NewZealand',
        'RGI19_SubantarcticAntarctic'
    ]
    
    # Satellite evaluation regions
    SATELLITE_REGIONS = ['R1', 'R2']
    SATELLITE_YEARS = ['2020', '2021', '2022', '2023', '2024', '2025']
    
    # Model configuration
    MODEL_TYPE = 'timesformer'
    IMAGE_SIZE = 128
    NUM_FRAMES = 6
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 4
    
    # Training configuration
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 80
    WARMUP_EPOCHS = 5
    WEIGHT_DECAY = 1e-4
    
    # Loss weights
    DICE_WEIGHT = 0.5
    BCE_WEIGHT = 0.3
    FOCAL_WEIGHT = 0.2
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    
    # Data augmentation
    AUGMENTATION = True
    ROTATION_RANGE = 15
    FLIP_PROBABILITY = 0.5
    BRIGHTNESS_RANGE = 0.2
    
    # Mixed precision training
    USE_AMP = True
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 15
    MIN_DELTA = 1e-4
    
    # GLOF risk detection
    GLOF_AREA_THRESHOLD = 0.20
    GLOF_VELOCITY_THRESHOLD = 1.5
    
    # Device configuration
    DEVICE = 'cuda'
    NUM_WORKERS = 0  # Windows compatibility
    PIN_MEMORY = True
    
    # Logging
    LOG_INTERVAL = 10
    SAVE_INTERVAL = 5
    
    # Target generation parameters
    TARGET_MODE = 'relaxed'  # Changed to relaxed
    VELOCITY_PERCENTILE = 75
    DEM_DEPRESSION_THRESHOLD = -50
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        dirs = [
            cls.FEATURES_DIR, cls.STATIC_FEATURES_DIR, cls.DYNAMIC_FEATURES_DIR,
            cls.MODELS_DIR, cls.CHECKPOINT_DIR, cls.ARCHIVE_DIR,
            cls.EVAL_DIR, cls.METRICS_DIR, cls.PLOTS_DIR
        ]
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_satellite_paths(cls, year, region, sensor='Sentinel2'):
        """Get satellite image paths for evaluation"""
        sensor_dir = cls.SENTINEL2_DIR if sensor == 'Sentinel2' else cls.SENTINEL1_DIR
        region_dir = sensor_dir / year / region
        
        if not region_dir.exists():
            return []
        
        return sorted(region_dir.glob('*.tiff')) + sorted(region_dir.glob('*.tif'))
    
    @classmethod
    def get_mass_balance_path(cls, region):
        """Get mass balance data path for a region"""
        regional_dir = cls.MASS_BALANCE_DIR / "Regional"
        region_clean = region.lower().replace('_', '')
        return regional_dir / f"{region_clean}_mass_balance.csv"

Config.create_directories()
