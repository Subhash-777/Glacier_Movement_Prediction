"""
Target generation for glacier lake prediction
Creates ground truth masks based on velocity patterns and DEM
"""
import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion, label, zoom

class TargetGenerator:
    
    def __init__(self, mode='relaxed'):
        """
        mode: 'strict', 'balanced', or 'relaxed'
        """
        self.mode = mode
        
    def generate_lake_targets(self, velocity_mag, dem, depressions=None):
        """
        Generate glacier lake targets from velocity and DEM
        More lenient approach for small datasets
        """
        # Ensure same shape
        if velocity_mag.shape != dem.shape:
            if dem.shape != velocity_mag.shape:
                zoom_factor = (velocity_mag.shape[0] / dem.shape[0], 
                              velocity_mag.shape[1] / dem.shape[1])
                dem = zoom(dem, zoom_factor, order=1)
            if depressions is not None and depressions.shape != velocity_mag.shape:
                zoom_factor = (velocity_mag.shape[0] / depressions.shape[0], 
                              velocity_mag.shape[1] / depressions.shape[1])
                depressions = zoom(depressions, zoom_factor, order=1)
        
        # Normalize inputs
        velocity_norm = self._normalize(velocity_mag)
        dem_norm = self._normalize(dem)
        
        # Initialize target
        target = np.zeros_like(velocity_mag, dtype=np.float32)
        
        # Method 1: Low velocity regions
        velocity_threshold = self._get_velocity_threshold(velocity_mag)
        low_velocity_mask = velocity_mag < velocity_threshold
        
        # Method 2: DEM depressions
        if depressions is not None and np.any(depressions > 0):
            depression_mask = depressions > 0
        else:
            depression_mask = self._detect_depressions(dem)
        
        # Method 3: Velocity divergence
        dvx_dx = np.gradient(velocity_mag, axis=1)
        dvy_dy = np.gradient(velocity_mag, axis=0)
        divergence = dvx_dx + dvy_dy
        convergence_mask = divergence < np.percentile(divergence, 30)
        
        # Combine criteria based on mode
        if self.mode == 'strict':
            target = (low_velocity_mask & depression_mask & convergence_mask).astype(np.float32)
            
        elif self.mode == 'balanced':
            combined = low_velocity_mask.astype(int) + depression_mask.astype(int) + convergence_mask.astype(int)
            target = (combined >= 2).astype(np.float32)
            
        elif self.mode == 'relaxed':
            target = (low_velocity_mask | depression_mask).astype(np.float32)
            
            # If still no targets, create some
            if np.sum(target) < 10:
                target = (velocity_mag < np.percentile(velocity_mag, 5)).astype(np.float32)
        
        # Post-processing
        target = self._postprocess_target(target, min_size=20)
        
        return target
    
    def generate_glof_risk_targets(self, velocity_sequence, area_sequence):
        """Generate GLOF risk targets"""
        glof_risk = np.zeros(velocity_sequence.shape[1:], dtype=np.float32)
        
        if len(area_sequence) > 1:
            area_changes = np.diff(area_sequence) / (area_sequence[:-1] + 1e-8)
            if np.any(area_changes > 0.20):
                glof_risk += 0.5
        
        if velocity_sequence.shape[0] > 1:
            velocity_trend = velocity_sequence[-1] - velocity_sequence[0]
            high_acceleration = velocity_trend > np.percentile(velocity_trend, 90)
            glof_risk[high_acceleration] += 0.5
        
        glof_risk = np.clip(glof_risk, 0, 1)
        return glof_risk
    
    def _get_velocity_threshold(self, velocity_mag):
        """Determine velocity threshold"""
        if self.mode == 'strict':
            return np.percentile(velocity_mag[velocity_mag > 0], 10)
        elif self.mode == 'balanced':
            return np.percentile(velocity_mag[velocity_mag > 0], 20)
        else:
            return np.percentile(velocity_mag[velocity_mag > 0], 30)
    
    def _detect_depressions(self, dem):
        """Simple depression detection"""
        footprint = np.ones((7, 7))
        local_min = ndimage.minimum_filter(dem, footprint=footprint)
        depressions = (dem == local_min) & (dem < np.percentile(dem, 30))
        return depressions
    
    def _normalize(self, array):
        """Min-max normalization"""
        array_min = np.min(array)
        array_max = np.max(array)
        if array_max - array_min < 1e-8:
            return np.zeros_like(array)
        return (array - array_min) / (array_max - array_min)
    
    def _postprocess_target(self, target, min_size=20):
        """Post-process target mask"""
        # Remove small regions
        labeled, num_features = label(target)
        for i in range(1, num_features + 1):
            region = labeled == i
            if np.sum(region) < min_size:
                target[region] = 0
        
        # Light smoothing
        if np.sum(target) > 0:
            target = binary_erosion(target, structure=np.ones((2, 2)))
            target = binary_dilation(target, structure=np.ones((2, 2)))
        
        return target.astype(np.float32)
