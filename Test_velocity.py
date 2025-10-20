"""
Test script to inspect ITS_LIVE NetCDF structure
"""
import xarray as xr
import netCDF4 as nc
from pathlib import Path

def inspect_netcdf(filepath):
    """Inspect NetCDF file structure"""
    print(f"\n{'='*60}")
    print(f"Inspecting: {filepath.name}")
    print(f"{'='*60}\n")
    
    try:
        # Method 1: xarray
        print("Opening with xarray...")
        ds = xr.open_dataset(filepath, engine='netcdf4')
        
        print(f"\nDimensions: {dict(ds.dims)}")
        print(f"\nVariables:")
        for var in ds.variables:
            print(f"  - {var}: {ds[var].shape} ({ds[var].dtype})")
        
        print(f"\nCoordinates:")
        for coord in ds.coords:
            print(f"  - {coord}: {ds.coords[coord].shape}")
        
        print(f"\nAttributes:")
        for attr in ds.attrs:
            print(f"  - {attr}: {ds.attrs[attr]}")
        
        ds.close()
        return True
        
    except Exception as e1:
        print(f"xarray failed: {e1}")
        
        try:
            # Method 2: netCDF4
            print("\nTrying netCDF4...")
            dataset = nc.Dataset(filepath, 'r')
            
            print(f"\nDimensions: {dataset.dimensions}")
            print(f"\nVariables:")
            for var in dataset.variables:
                print(f"  - {var}: {dataset.variables[var].shape}")
            
            print(f"\nAttributes:")
            for attr in dataset.ncattrs():
                print(f"  - {attr}: {getattr(dataset, attr)}")
            
            dataset.close()
            return True
            
        except Exception as e2:
            print(f"netCDF4 also failed: {e2}")
            return False

# Test on first velocity file
velocity_dir = Path("data/velocity/ITS_LIVE/RGI01_Alaska")
velocity_files = list(velocity_dir.glob("*.nc"))

if velocity_files:
    inspect_netcdf(velocity_files[0])
else:
    print("No .nc files found!")
