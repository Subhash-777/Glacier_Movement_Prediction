# Glacier Movement Prediction System
**Bharatiya Antariksh Hackathon 2025**

A deep learning system for predicting glacier lake formation and GLOF (Glacial Lake Outburst Flood) risks using multi-modal satellite data and velocity measurements.

## Features

- **Multi-Region Support**: 13 RGI regions including Alaska, Karakoram, Iceland, and more
- **Multi-Modal Data**: ITS_LIVE velocity, DEM (SRTM/ASTER/CartoDEM), Sentinel-1/2 imagery
- **Two Model Architectures**:
  - Enhanced TimeSformer (video transformer with divided space-time attention)
  - Simple 3D CNN (better for small datasets)
- **Memory Optimized**: Runs on 4GB GPU with mixed precision training
- **Temporal Analysis**: GLOF risk detection based on area changes
- **Satellite Evaluation**: Direct evaluation on Sentinel-2 imagery

## Dataset Structure
GlacierMovementPrediction/
├── data/
│ ├── velocity/ITS_LIVE/ # Velocity data (13 regions)
│ ├── dem/ # SRTM, ASTER_GDEM, CartoDEM
│ ├── outlines/ # RGI glacier outlines
│ ├── centerlines/ # Glacier centerlines
│ ├── satellite/
│ │ ├── Sentinel1/ # C-band SAR
│ │ ├── Sentinel2/ # Optical (2020-2025, R1-R5)
│ │ └── Landsat/ # For future expansion
│ └── mass_balance/ # WGMS mass balance data
├── features/ # Extracted features
├── models/checkpoints/ # Trained models
└── evaluation/ # Results and visualizations

text

## Installation

Clone repository
git clone <repository-url>
Anaconda Env Create

conda create -n glacier_pred python=3.10 -y && conda activate glacier_pred 

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y && conda install -c conda-forge gdal=3.4 rasterio geopandas shapely fiona pyproj xarray netcdf4 h5py zarr -y && conda install numpy scipy pandas scikit-learn scikit-image opencv matplotlib seaborn plotly -y
pip install albumentations imgaug pillow fastapi "uvicorn[standard]" streamlit pydantic python-multipart aiofiles tqdm pyyaml wandb tensorboard pytest pytest-cov black flake8 mypy accelerate onnx onnxruntime requests boto3 google-cloud-storage


text

## Usage

### 1. Feature Extraction

Extract features from raw data:

python main.py --mode extract

text

Extract specific regions:

python main.py --mode extract --regions RGI01_Alaska RGI14_Karakoram

text

### 2. Training

Train with TimeSformer:

python main.py --mode train --model_type timesformer --epochs 80

text    

Train with Simple 3D CNN (recommended for small datasets):

python main.py --mode train --model_type simple_cnn --epochs 50

text

Custom hyperparameters:

python main.py --mode train --model_type timesformer --epochs 100 --batch_size 2 --learning_rate 1e-4

text

### 3. Evaluation

Evaluate on test regions:

python main.py --mode evaluate

text

### 4. Satellite Evaluation

Evaluate on Sentinel-2 imagery:

python main.py --mode satellite_eval

text

### 5. Full Pipeline

Run complete pipeline:

python main.py --mode all

text

## Model Architecture

### Enhanced TimeSformer
- **Parameters**: 3.7M (optimized for 4GB GPU)
- **Input**: (B, T=6, C=8, H=128, W=128)
- **Features**: Divided space-time attention, DEM fusion, change detection
- **Best for**: Larger datasets (>100 samples)

### Simple 3D CNN
- **Parameters**: 2.1M
- **Architecture**: 3D encoder-decoder with residual blocks
- **Best for**: Small datasets (<50 samples)

## Configuration

Edit `scripts/utils/config.py` to customize:

Model settings
IMAGE_SIZE = 128 # Image size
NUM_FRAMES = 6 # Temporal frames
BATCH_SIZE = 1 # Batch size
MODEL_TYPE = 'timesformer' # or 'simple_cnn'

Training settings
LEARNING_RATE = 5e-5
NUM_EPOCHS = 80
USE_AMP = True # Mixed precision

Loss weights
DICE_WEIGHT = 0.5
BCE_WEIGHT = 0.3
FOCAL_WEIGHT = 0.2

GLOF detection
GLOF_AREA_THRESHOLD = 0.20 # 20% area change

text

## Results

Expected performance metrics:
- **IoU**: 0.30 - 0.50
- **Dice**: 0.40 - 0.65
- **F1 Score**: 0.40 - 0.65
- **Precision**: 0.50 - 0.75

## Satellite Regions

Currently supports:
- **R1, R2**: Sentinel-2 data (2020-2025)
- **Expandable to R3, R4, R5**: Add data to respective folders

## GLOF Risk Detection

Detects GLOF risk based on:
- Rapid area increase (>20% threshold)
- Temporal velocity trends
- Multi-year lake evolution

## Troubleshooting

### Out of Memory
- Reduce `IMAGE_SIZE` (128 → 96)
- Increase `GRADIENT_ACCUMULATION_STEPS`
- Use `simple_cnn` model

### Low Metrics
- Check target generation mode (`balanced` recommended)
- Ensure proper checkpoint loading
- Verify data quality and alignment

### Missing Data
- Run feature extraction first
- Check data folder structure
- Verify file formats (.tif for DEM, .nc/.tif for velocity)

## Citation

If you use this code, please cite:

@software{glacier_movement_prediction_2025,
title={Glacier Movement Prediction System},
author={[Your Name]},
year={2025},
note={Bharatiya Antariksh Hackathon 2025}
}

text

## License

MIT License

## Contact

For questions or issues, please contact [your-email]
