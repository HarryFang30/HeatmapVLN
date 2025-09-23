"""
Custom exceptions for VLN Project
Provides specific error types for better error handling and debugging
"""

class VLNError(Exception):
    """Base exception class for VLN-related errors"""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "VLN_GENERIC_ERROR"
        self.details = details or {}
    
    def __str__(self):
        base_str = f"[{self.error_code}] {self.message}"
        if self.details:
            details_str = ", ".join([f"{k}={v}" for k, v in self.details.items()])
            return f"{base_str} (Details: {details_str})"
        return base_str


class ConfigurationError(VLNError):
    """Raised when configuration is invalid or missing"""
    
    def __init__(self, message: str, config_key: str = None, expected_value=None):
        details = {}
        if config_key:
            details["config_key"] = config_key
        if expected_value:
            details["expected_value"] = expected_value
        
        super().__init__(
            message=message,
            error_code="VLN_CONFIG_ERROR",
            details=details
        )


class VideoProcessingError(VLNError):
    """Raised when video processing fails"""
    
    def __init__(self, message: str, video_path: str = None, frame_index: int = None):
        details = {}
        if video_path:
            details["video_path"] = video_path
        if frame_index is not None:
            details["frame_index"] = frame_index
            
        super().__init__(
            message=message,
            error_code="VLN_VIDEO_ERROR",
            details=details
        )


class SamplingError(VLNError):
    """Raised when frame sampling fails"""
    
    def __init__(self, message: str, total_frames: int = None, target_frames: int = None):
        details = {}
        if total_frames is not None:
            details["total_frames"] = total_frames
        if target_frames is not None:
            details["target_frames"] = target_frames
            
        super().__init__(
            message=message,
            error_code="VLN_SAMPLING_ERROR", 
            details=details
        )


class ModelError(VLNError):
    """Raised when model operations fail"""
    
    def __init__(self, message: str, model_name: str = None, operation: str = None):
        details = {}
        if model_name:
            details["model_name"] = model_name
        if operation:
            details["operation"] = operation
            
        super().__init__(
            message=message,
            error_code="VLN_MODEL_ERROR",
            details=details
        )


class HeatmapError(VLNError):
    """Raised when heatmap generation fails"""
    
    def __init__(self, message: str, heatmap_size: tuple = None, num_views: int = None):
        details = {}
        if heatmap_size:
            details["heatmap_size"] = heatmap_size
        if num_views is not None:
            details["num_views"] = num_views
            
        super().__init__(
            message=message,
            error_code="VLN_HEATMAP_ERROR",
            details=details
        )


class DataLoaderError(VLNError):
    """Raised when data loading fails"""
    
    def __init__(self, message: str, dataset_path: str = None, batch_index: int = None):
        details = {}
        if dataset_path:
            details["dataset_path"] = dataset_path
        if batch_index is not None:
            details["batch_index"] = batch_index
            
        super().__init__(
            message=message,
            error_code="VLN_DATA_ERROR",
            details=details
        )


class ValidationError(VLNError):
    """Raised when input validation fails"""
    
    def __init__(self, message: str, parameter_name: str = None, received_value=None, expected_type=None):
        details = {}
        if parameter_name:
            details["parameter_name"] = parameter_name
        if received_value is not None:
            details["received_value"] = str(received_value)
        if expected_type:
            details["expected_type"] = str(expected_type)
            
        super().__init__(
            message=message,
            error_code="VLN_VALIDATION_ERROR",
            details=details
        )


class ResourceError(VLNError):
    """Raised when system resources are insufficient"""
    
    def __init__(self, message: str, resource_type: str = None, required_amount=None, available_amount=None):
        details = {}
        if resource_type:
            details["resource_type"] = resource_type
        if required_amount is not None:
            details["required_amount"] = str(required_amount)
        if available_amount is not None:
            details["available_amount"] = str(available_amount)
            
        super().__init__(
            message=message,
            error_code="VLN_RESOURCE_ERROR",
            details=details
        )


class GeometryError(VLNError):
    """Raised when geometry processing fails"""
    
    def __init__(self, message: str, geometry_type: str = None, shape=None):
        details = {}
        if geometry_type:
            details["geometry_type"] = geometry_type
        if shape is not None:
            details["shape"] = str(shape)
            
        super().__init__(
            message=message,
            error_code="VLN_GEOMETRY_ERROR",
            details=details
        )


class FeatureFusionError(VLNError):
    """Raised when feature fusion fails"""
    
    def __init__(self, message: str, feature1_shape=None, feature2_shape=None, fusion_method: str = None):
        details = {}
        if feature1_shape is not None:
            details["feature1_shape"] = str(feature1_shape)
        if feature2_shape is not None:
            details["feature2_shape"] = str(feature2_shape)
        if fusion_method:
            details["fusion_method"] = fusion_method
            
        super().__init__(
            message=message,
            error_code="VLN_FUSION_ERROR",
            details=details
        )


class CheckpointError(VLNError):
    """Raised when checkpoint operations fail"""
    
    def __init__(self, message: str, checkpoint_path: str = None, operation: str = None):
        details = {}
        if checkpoint_path:
            details["checkpoint_path"] = checkpoint_path
        if operation:
            details["operation"] = operation
            
        super().__init__(
            message=message,
            error_code="VLN_CHECKPOINT_ERROR",
            details=details
        )


class EvaluationError(VLNError):
    """Raised when evaluation fails"""
    
    def __init__(self, message: str, metric_name: str = None, benchmark: str = None):
        details = {}
        if metric_name:
            details["metric_name"] = metric_name
        if benchmark:
            details["benchmark"] = benchmark
            
        super().__init__(
            message=message,
            error_code="VLN_EVALUATION_ERROR",
            details=details
        )