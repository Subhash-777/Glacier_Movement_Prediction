"""
Feature extraction for glacier movement prediction
Memory-optimized version - downsamples data during extraction
"""
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm
import pickle
from scripts.utils.config import Config
from scripts.utils.data_utils import GlacierDataUtils

class FeatureExtractor:
    
    def __init__(self, region, config=Config, target_size=512):
        """
        Args:
            region: RGI region name
            config: Configuration object
            target_size: Downsample to this size to save memory (default 512x512)
        """
        self.region = region
        self.config = config
        self.utils = GlacierDataUtils()
        self.target_size = target_size  # Reduced from full resolution
        
        # Paths
        self.velocity_dir = config.VELOCITY_DIR / region
        self.dem_dir = config.DEM_DIR
        self.outlines_dir = config.OUTLINES_DIR / region
        
        # Output paths
        self.static_output = config.STATIC_FEATURES_DIR / f"{region}_static.pkl"
        self.dynamic_output = config.DYNAMIC_FEATURES_DIR / f"{region}_dynamic.pkl"
        
    def extract_static_features(self):
        """
        Extract static features: DEM, slope, aspect, outlines (downsampled)
        """
        print(f"Extracting static features for {self.region}...")
        
        static_features = {}
        
        # Load DEM (prefer SRTM, fallback to ASTER)
        srtm_path = self.dem_dir / "SRTM" / self.region
        aster_path = self.dem_dir / "ASTER_GDEM" / self.region
        carto_path = self.dem_dir / "CartoDEM" / self.region
        
        dem_path = None
        if srtm_path.exists():
            dem_files = list(srtm_path.glob("*.tif")) + list(srtm_path.glob("*.tiff"))
            if dem_files:
                dem_path = dem_files[0]
        
        if dem_path is None and aster_path.exists():
            dem_files = list(aster_path.glob("*.tif")) + list(aster_path.glob("*.tiff"))
            if dem_files:
                dem_path = dem_files[0]
                
        if dem_path is None and carto_path.exists():
            dem_files = list(carto_path.glob("*.tif")) + list(carto_path.glob("*.tiff"))
            if dem_files:
                dem_path = dem_files[0]
        
        if dem_path:
            try:
                dem, transform, crs = self.utils.load_dem_data(dem_path)
                
                # Downsample DEM to save memory
                original_shape = dem.shape
                dem = self.utils.resize_array(dem, (self.target_size, self.target_size))
                
                static_features['dem'] = dem
                static_features['original_shape'] = original_shape
                static_features['transform'] = transform
                static_features['crs'] = crs
                
                # Compute slope and aspect
                slope, aspect = self.utils.compute_slope_aspect(dem)
                static_features['slope'] = slope
                static_features['aspect'] = aspect
                
                # Detect depressions
                depressions = self.utils.detect_depressions(dem)
                static_features['depressions'] = depressions
                
                print(f"  ✓ DEM loaded: {original_shape} -> {dem.shape}")
            except Exception as e:
                print(f"  ✗ Error loading DEM: {e}")
        else:
            print(f"  ✗ No DEM found for {self.region}")
            # Create dummy static features
            static_features['dem'] = np.zeros((self.target_size, self.target_size))
            static_features['slope'] = np.zeros((self.target_size, self.target_size))
            static_features['aspect'] = np.zeros((self.target_size, self.target_size))
            static_features['depressions'] = np.zeros((self.target_size, self.target_size))
            
        # Load glacier outlines (just metadata, not full geometry)
        if self.outlines_dir.exists():
            outline_files = list(self.outlines_dir.glob("*.shp"))
            if outline_files:
                try:
                    outlines = self.utils.load_glacier_outline(outline_files[0])
                    # Store only count and bounds, not full geometry
                    static_features['outline_count'] = len(outlines)
                    static_features['outline_bounds'] = outlines.total_bounds
                    print(f"  ✓ Outlines loaded: {len(outlines)} glaciers")
                except Exception as e:
                    print(f"  ✗ Error loading outlines: {e}")
            else:
                print(f"  ✗ No outlines found for {self.region}")
        
        # Save
        with open(self.static_output, 'wb') as f:
            pickle.dump(static_features, f)
            
        print(f"  ✓ Static features saved to {self.static_output}\n")
        return static_features
    
    def extract_dynamic_features(self):
        """
        Extract dynamic features: velocity time series (downsampled and compressed)
        """
        print(f"Extracting dynamic features for {self.region}...")
        
        dynamic_features = []
        
        if not self.velocity_dir.exists():
            print(f"  ✗ No velocity data found for {self.region}")
            return dynamic_features
        
        # Get all velocity files (sorted by time)
        velocity_files = sorted(self.velocity_dir.glob("*.nc"))
        
        if not velocity_files:
            velocity_files = sorted(self.velocity_dir.glob("*.tif"))
        
        print(f"  Found {len(velocity_files)} velocity files")
        
        successful_loads = 0
        for vel_file in tqdm(velocity_files, desc="  Processing velocity"):
            try:
                result = self.utils.load_velocity_data(vel_file)
                
                # Check if loading failed
                if result[0] is None:
                    continue
                
                vx, vy, v_mag, transform, crs = result
                
                # Downsample to save memory
                original_shape = vx.shape
                vx = self.utils.resize_array(vx, (self.target_size, self.target_size))
                vy = self.utils.resize_array(vy, (self.target_size, self.target_size))
                v_mag = self.utils.resize_array(v_mag, (self.target_size, self.target_size))
                
                # Compute additional features
                divergence = self.utils.compute_velocity_divergence(vx, vy)
                
                # Store as float32 to save memory
                feature_dict = {
                    'vx': vx.astype(np.float32),
                    'vy': vy.astype(np.float32),
                    'v_magnitude': v_mag.astype(np.float32),
                    'divergence': divergence.astype(np.float32),
                    'original_shape': original_shape,
                    'filename': vel_file.name
                }
                
                dynamic_features.append(feature_dict)
                successful_loads += 1
                
                # Clear memory
                del vx, vy, v_mag, divergence
                
            except Exception as e:
                print(f"    ✗ Error loading {vel_file.name}: {e}")
                continue
        
        print(f"  ✓ Successfully loaded {successful_loads}/{len(velocity_files)} velocity files")
        
        # Save with compression
        print(f"  Saving features (this may take a moment)...")
        with open(self.dynamic_output, 'wb') as f:
            pickle.dump(dynamic_features, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        # Report file size
        file_size_mb = self.dynamic_output.stat().st_size / (1024 * 1024)
        print(f"  ✓ Dynamic features saved: {file_size_mb:.1f} MB\n")
        
        return dynamic_features
    
    def extract_all(self):
        """Extract both static and dynamic features"""
        static = self.extract_static_features()
        dynamic = self.extract_dynamic_features()
        return static, dynamic

def extract_all_regions(regions=None, target_size=512):
    """
    Extract features for all regions with memory optimization
    
    Args:
        regions: List of regions to process
        target_size: Downsample size (512x512 by default)
    """
    if regions is None:
        regions = Config.RGI_REGIONS
    
    print(f"\n{'='*60}")
    print(f"FEATURE EXTRACTION FOR {len(regions)} REGIONS")
    print(f"Target size: {target_size}x{target_size} (memory optimized)")
    print(f"{'='*60}\n")
    
    results_summary = {}
    
    for region in regions:
        try:
            extractor = FeatureExtractor(region, target_size=target_size)
            static, dynamic = extractor.extract_all()
            
            results_summary[region] = {
                'has_dem': 'dem' in static,
                'has_outlines': 'outline_count' in static,
                'num_velocity_files': len(dynamic),
                'status': 'success'
            }
        except Exception as e:
            print(f"  ✗ Failed to extract features for {region}: {e}\n")
            results_summary[region] = {
                'has_dem': False,
                'has_outlines': False,
                'num_velocity_files': 0,
                'status': f'failed: {e}'
            }
            continue
    
    print(f"\n{'='*60}")
    print("FEATURE EXTRACTION COMPLETE")
    print(f"{'='*60}\n")
    
    print("Summary:")
    for region, info in results_summary.items():
        status_icon = '✓' if info['status'] == 'success' else '✗'
        print(f"  {status_icon} {region}:")
        if info['status'] == 'success':
            print(f"      DEM: {'✓' if info['has_dem'] else '✗'}")
            print(f"      Outlines: {'✓' if info['has_outlines'] else '✗'}")
            print(f"      Velocity files: {info['num_velocity_files']}")
        else:
            print(f"      Status: {info['status']}")

if __name__ == "__main__":
    extract_all_regions()
