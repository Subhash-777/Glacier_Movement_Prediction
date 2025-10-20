"""
Check data quality across all regions
"""
import pickle
from pathlib import Path
from scripts.utils.config import Config
import numpy as np

def check_region_quality(region):
    """Check data quality for a region"""
    static_path = Config.STATIC_FEATURES_DIR / f"{region}_static.pkl"
    dynamic_path = Config.DYNAMIC_FEATURES_DIR / f"{region}_dynamic.pkl"
    
    info = {
        'region': region,
        'has_static': static_path.exists(),
        'has_dynamic': dynamic_path.exists(),
        'has_dem': False,
        'num_velocity_files': 0,
        'velocity_shape': None,
        'dem_shape': None
    }
    
    if static_path.exists():
        try:
            with open(static_path, 'rb') as f:
                static = pickle.load(f)
            info['has_dem'] = 'dem' in static and static['dem'] is not None
            if info['has_dem']:
                info['dem_shape'] = static['dem'].shape
        except Exception as e:
            print(f"Error loading static for {region}: {e}")
    
    if dynamic_path.exists():
        try:
            with open(dynamic_path, 'rb') as f:
                dynamic = pickle.load(f)
            info['num_velocity_files'] = len(dynamic)
            if len(dynamic) > 0:
                info['velocity_shape'] = dynamic[0]['v_magnitude'].shape
        except Exception as e:
            print(f"Error loading dynamic for {region}: {e}")
    
    return info

if __name__ == "__main__":
    print("\n" + "="*80)
    print("DATA QUALITY REPORT")
    print("="*80 + "\n")

    all_info = []
    for region in Config.RGI_REGIONS:
        info = check_region_quality(region)
        all_info.append(info)
        
        status = "✓" if info['has_static'] and info['has_dynamic'] and info['num_velocity_files'] >= 6 else "✗"
        
        print(f"{status} {region}")
        print(f"    DEM: {'✓' if info['has_dem'] else '✗'} {info['dem_shape'] if info['has_dem'] else 'N/A'}")
        print(f"    Velocity files: {info['num_velocity_files']} {info['velocity_shape'] if info['velocity_shape'] else 'N/A'}")
        print()

    # Suggest best train/val split
    good_regions = [info for info in all_info if info['num_velocity_files'] >= 6]
    print("="*80)
    print(f"RECOMMENDATION: {len(good_regions)} regions have sufficient data (≥6 velocity files)")
    print("="*80 + "\n")

    if len(good_regions) >= 2:
        print("Suggested train/val split:")
        train_count = max(1, int(len(good_regions) * 0.8))
        train_regions = [r['region'] for r in good_regions[:train_count]]
        val_regions = [r['region'] for r in good_regions[train_count:]]
        
        print(f"\nTrain regions ({len(train_regions)}):")
        for r in train_regions:
            print(f"  - {r}")
        
        print(f"\nValidation regions ({len(val_regions)}):")
        for r in val_regions:
            print(f"  - {r}")
    else:
        print("Warning: Not enough regions with sufficient data for training!")
