"""
Data utility functions for glacier data processing
Updated to handle ITS_LIVE NetCDF format
"""
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
import geopandas as gpd
from pathlib import Path
import warnings
import xarray as xr
import netCDF4 as nc
warnings.filterwarnings('ignore')

class GlacierDataUtils:
    
    @staticmethod
    def load_velocity_data(velocity_path, bounds=None):
        """
        Load ITS_LIVE velocity data from NetCDF format
        Returns: vx, vy, v_magnitude, transform, crs
        """
        velocity_path = str(velocity_path)
        
        try:
            # Try with xarray first (recommended for ITS_LIVE data)
            ds = xr.open_dataset(velocity_path, engine='netcdf4')
            
            # ITS_LIVE variable names
            # Common variable names: 'v', 'vx', 'vy', 'v_error'
            if 'vx' in ds.variables and 'vy' in ds.variables:
                vx = ds['vx'].values
                vy = ds['vy'].values
            elif 'VX' in ds.variables and 'VY' in ds.variables:
                vx = ds['VX'].values
                vy = ds['VY'].values
            elif 'v_x' in ds.variables and 'v_y' in ds.variables:
                vx = ds['v_x'].values
                vy = ds['v_y'].values
            else:
                # Try to find velocity variables
                var_names = list(ds.variables.keys())
                print(f"    Warning: Standard velocity variables not found. Available: {var_names}")
                ds.close()
                raise ValueError("Could not find velocity variables")
            
            # Get coordinate information
            if 'x' in ds.coords and 'y' in ds.coords:
                x = ds['x'].values
                y = ds['y'].values
            elif 'lon' in ds.coords and 'lat' in ds.coords:
                x = ds['lon'].values
                y = ds['lat'].values
            else:
                x = np.arange(vx.shape[1])
                y = np.arange(vx.shape[0])
            
            # Create affine transform
            from rasterio.transform import from_bounds
            if len(x) > 1 and len(y) > 1:
                pixel_size_x = np.abs(x[1] - x[0])
                pixel_size_y = np.abs(y[1] - y[0])
                
                transform = rasterio.Affine(
                    pixel_size_x, 0, x.min(),
                    0, -pixel_size_y, y.max()
                )
            else:
                transform = rasterio.Affine.identity()
            
            # Get CRS
            if 'mapping' in ds.variables:
                mapping = ds['mapping']
                if hasattr(mapping, 'spatial_ref'):
                    crs = rasterio.crs.CRS.from_wkt(mapping.spatial_ref)
                elif hasattr(mapping, 'crs_wkt'):
                    crs = rasterio.crs.CRS.from_wkt(mapping.crs_wkt)
                else:
                    crs = rasterio.crs.CRS.from_epsg(3413)  # Default to NSIDC
            elif 'crs' in ds.variables:
                crs_var = ds['crs']
                if hasattr(crs_var, 'spatial_ref'):
                    crs = rasterio.crs.CRS.from_wkt(crs_var.spatial_ref)
                else:
                    crs = rasterio.crs.CRS.from_epsg(3413)
            else:
                # Default CRS for polar regions
                crs = rasterio.crs.CRS.from_epsg(3413)
            
            ds.close()
            
        except Exception as e:
            # Fallback to netCDF4
            try:
                dataset = nc.Dataset(velocity_path, 'r')
                
                # Try different variable name conventions
                if 'vx' in dataset.variables:
                    vx = dataset.variables['vx'][:]
                    vy = dataset.variables['vy'][:]
                elif 'VX' in dataset.variables:
                    vx = dataset.variables['VX'][:]
                    vy = dataset.variables['VY'][:]
                elif 'v_x' in dataset.variables:
                    vx = dataset.variables['v_x'][:]
                    vy = dataset.variables['v_y'][:]
                else:
                    var_names = list(dataset.variables.keys())
                    raise ValueError(f"Could not find velocity variables. Available: {var_names}")
                
                # Get coordinates
                if 'x' in dataset.variables:
                    x = dataset.variables['x'][:]
                    y = dataset.variables['y'][:]
                elif 'lon' in dataset.variables:
                    x = dataset.variables['lon'][:]
                    y = dataset.variables['lat'][:]
                else:
                    x = np.arange(vx.shape[1] if vx.ndim > 1 else len(vx))
                    y = np.arange(vx.shape[0] if vx.ndim > 1 else 1)
                
                # Create transform
                if len(x) > 1 and len(y) > 1:
                    pixel_size_x = np.abs(x[1] - x[0])
                    pixel_size_y = np.abs(y[1] - y[0])
                    
                    transform = rasterio.Affine(
                        pixel_size_x, 0, x.min(),
                        0, -pixel_size_y, y.max()
                    )
                else:
                    transform = rasterio.Affine.identity()
                
                crs = rasterio.crs.CRS.from_epsg(3413)
                
                dataset.close()
                
            except Exception as e2:
                raise ValueError(f"Failed to load velocity data: {e2}")
        
        # Handle masked arrays
        if hasattr(vx, 'mask'):
            vx = np.ma.filled(vx, 0.0)
        if hasattr(vy, 'mask'):
            vy = np.ma.filled(vy, 0.0)
        
        # Ensure 2D arrays
        if vx.ndim > 2:
            vx = vx.squeeze()
        if vy.ndim > 2:
            vy = vy.squeeze()
        
        # Handle 1D or 0D arrays
        if vx.ndim < 2:
            print(f"    Warning: Velocity data has unexpected dimensions: {vx.ndim}D")
            return None, None, None, None, None
        
        # Calculate magnitude
        v_magnitude = np.sqrt(vx**2 + vy**2)
        
        # Replace invalid values
        vx = np.nan_to_num(vx, nan=0.0, posinf=0.0, neginf=0.0)
        vy = np.nan_to_num(vy, nan=0.0, posinf=0.0, neginf=0.0)
        v_magnitude = np.nan_to_num(v_magnitude, nan=0.0, posinf=0.0, neginf=0.0)
        
        return vx, vy, v_magnitude, transform, crs
    
    @staticmethod
    def load_dem_data(dem_path, target_shape=None, target_transform=None):
        """
        Load DEM data and optionally reproject to target shape
        """
        with rasterio.open(dem_path) as src:
            dem = src.read(1)
            transform = src.transform
            crs = src.crs
            
        # Handle nodata values
        dem = np.nan_to_num(dem, nan=0.0)
        
        # Reproject if needed
        if target_shape is not None and target_transform is not None:
            dem_reprojected = np.zeros(target_shape, dtype=np.float32)
            reproject(
                source=dem,
                destination=dem_reprojected,
                src_transform=transform,
                dst_transform=target_transform,
                src_crs=crs,
                dst_crs=crs,
                resampling=Resampling.bilinear
            )
            dem = dem_reprojected
            transform = target_transform
            
        return dem, transform, crs
    
    @staticmethod
    def load_glacier_outline(outline_path, bbox=None):
        """
        Load glacier outline shapefile
        """
        gdf = gpd.read_file(outline_path)
        
        if bbox:
            # Filter by bounding box
            gdf = gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            
        return gdf
    
    @staticmethod
    def load_satellite_image(satellite_path):
        """
        Load satellite imagery (Sentinel-1/2, Landsat)
        Returns: image array, metadata
        """
        with rasterio.open(satellite_path) as src:
            # Read all bands
            image = src.read()
            meta = src.meta.copy()
            transform = src.transform
            crs = src.crs
            
        # Handle nodata
        image = np.nan_to_num(image, nan=0.0)
        
        return image, meta, transform, crs
    
    @staticmethod
    def normalize_data(data, method='minmax', percentile_range=(2, 98)):
        """
        Normalize data for neural network input
        """
        if method == 'minmax':
            data_min = np.percentile(data, percentile_range[0])
            data_max = np.percentile(data, percentile_range[1])
            normalized = (data - data_min) / (data_max - data_min + 1e-8)
            normalized = np.clip(normalized, 0, 1)
            
        elif method == 'zscore':
            mean = np.mean(data)
            std = np.std(data)
            normalized = (data - mean) / (std + 1e-8)
            
        elif method == 'log':
            normalized = np.log1p(np.abs(data)) * np.sign(data)
            
        else:
            normalized = data
            
        return normalized.astype(np.float32)
    
    @staticmethod
    def compute_slope_aspect(dem):
        """
        Compute slope and aspect from DEM
        """
        # Compute gradients
        dy, dx = np.gradient(dem)
        
        # Slope in degrees
        slope = np.arctan(np.sqrt(dx**2 + dy**2)) * (180 / np.pi)
        
        # Aspect in degrees
        aspect = np.arctan2(-dx, dy) * (180 / np.pi)
        aspect = (aspect + 360) % 360
        
        return slope, aspect
    
    @staticmethod
    def detect_depressions(dem, threshold=-50):
        """
        Detect potential lake locations from DEM depressions
        """
        # Simple depression detection using morphological operations
        from scipy import ndimage
        
        # Invert DEM
        inverted = -dem
        
        # Local maxima in inverted DEM = depressions
        footprint = np.ones((5, 5))
        local_max = ndimage.maximum_filter(inverted, footprint=footprint)
        depressions = (inverted == local_max) & (dem < threshold)
        
        return depressions.astype(np.float32)
    
    @staticmethod
    def compute_velocity_divergence(vx, vy):
        """
        Compute velocity divergence (indicates potential accumulation zones)
        """
        dvx_dx = np.gradient(vx, axis=1)
        dvy_dy = np.gradient(vy, axis=0)
        divergence = dvx_dx + dvy_dy
        
        return divergence
    
    @staticmethod
    def resize_array(array, target_size):
        """
        Resize array to target size using interpolation
        """
        from scipy.ndimage import zoom
        
        if array.ndim == 2:
            zoom_factors = (target_size[0] / array.shape[0], 
                           target_size[1] / array.shape[1])
        elif array.ndim == 3:
            zoom_factors = (1, 
                           target_size[0] / array.shape[1], 
                           target_size[1] / array.shape[2])
        else:
            raise ValueError(f"Unsupported array dimension: {array.ndim}")
            
        resized = zoom(array, zoom_factors, order=1)
        return resized
    
    @staticmethod
    def create_temporal_sequence(data_list, num_frames=6):
        """
        Create temporal sequence from list of arrays
        """
        if len(data_list) < num_frames:
            # Pad with last frame
            while len(data_list) < num_frames:
                data_list.append(data_list[-1])
        elif len(data_list) > num_frames:
            # Sample uniformly
            indices = np.linspace(0, len(data_list)-1, num_frames, dtype=int)
            data_list = [data_list[i] for i in indices]
            
        return np.stack(data_list, axis=0)
    
    @staticmethod
    def load_mass_balance_data(csv_path):
        """
        Load mass balance data from CSV
        """
        import pandas as pd
        
        if not Path(csv_path).exists():
            return None
            
        df = pd.read_csv(csv_path)
        return df
