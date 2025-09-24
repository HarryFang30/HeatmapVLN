"""
Performance Optimizer for Dual-Encoder Architecture
==================================================

This module provides comprehensive performance optimizations for the VLN
dual-encoder architecture (VGGT + DINOv3). It implements memory-efficient
processing, compute optimizations, and intelligent caching strategies.

Key Optimizations:
1. Memory-efficient batch processing
2. Gradient checkpointing for large models  
3. Mixed precision training/inference
4. Intelligent feature caching
5. Multi-GPU support (when available)
6. Asynchronous processing pipelines
7. Model compilation optimizations

Performance Targets:
- Process 128 frames (N_m) → 16 keyframes (N_k) in <5 seconds on RTX 4090
- Memory usage <12GB for full pipeline
- Support batch sizes up to 4 for training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
import time
import gc
from contextlib import contextmanager
from dataclasses import dataclass
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class PerformanceConfig:
    """Configuration for performance optimizations."""
    # Memory optimizations
    enable_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True
    enable_memory_efficient_attention: bool = True
    clear_cache_frequency: int = 10  # Clear cache every N iterations
    
    # Compute optimizations
    compile_models: bool = True  # Use torch.compile
    use_channels_last: bool = True  # Memory layout optimization
    enable_async_processing: bool = True
    
    # Batch processing
    max_vggt_batch_size: int = 8  # Maximum frames per VGGT batch
    max_dinov3_batch_size: int = 4  # Maximum frames per DINOv3 batch
    adaptive_batch_size: bool = True  # Adjust batch size based on memory
    
    # Caching strategies
    enable_feature_caching: bool = True
    cache_vggt_features: bool = True
    cache_geometry_features: bool = True
    cache_size_limit_mb: int = 2048  # Maximum cache size in MB
    
    # Multi-GPU support
    enable_data_parallel: bool = False
    enable_model_parallel: bool = False
    
    # Debug and monitoring
    profile_memory_usage: bool = False
    profile_compute_time: bool = True
    log_optimization_stats: bool = True


class MemoryManager:
    """Manages memory usage and optimization for the dual-encoder pipeline."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.iteration_count = 0
        self.memory_stats = []
        
    @contextmanager
    def memory_efficient_context(self):
        """Context manager for memory-efficient processing."""
        try:
            # Enable memory-efficient settings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Set memory format to channels_last if enabled
            if self.config.use_channels_last:
                # This would be applied to tensors in the calling code
                pass
                
            yield
            
        finally:
            # Cleanup after processing
            self.iteration_count += 1
            
            if self.iteration_count % self.config.clear_cache_frequency == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
            if self.config.profile_memory_usage and torch.cuda.is_available():
                self._log_memory_usage()
    
    def _log_memory_usage(self):
        """Log current memory usage statistics."""
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        cached = torch.cuda.memory_reserved() / 1024**2  # MB
        
        self.memory_stats.append({
            'iteration': self.iteration_count,
            'allocated_mb': allocated,
            'cached_mb': cached
        })
        
        if self.iteration_count % 10 == 0:
            logger.info(f"Memory usage: {allocated:.1f}MB allocated, {cached:.1f}MB cached")
    
    def get_optimal_batch_size(
        self, 
        total_items: int, 
        item_memory_mb: float, 
        available_memory_mb: float
    ) -> int:
        """Calculate optimal batch size based on memory constraints."""
        
        if not self.config.adaptive_batch_size:
            return min(total_items, self.config.max_vggt_batch_size)
        
        # Leave 20% headroom for other operations
        usable_memory = available_memory_mb * 0.8
        
        # Calculate maximum batch size that fits in memory
        max_batch_size = int(usable_memory / (item_memory_mb + 1e-6))
        
        # Apply configured limits
        max_batch_size = min(max_batch_size, self.config.max_vggt_batch_size)
        max_batch_size = min(max_batch_size, total_items)
        
        return max(1, max_batch_size)


class ComputeOptimizer:
    """Optimizes compute operations for the dual-encoder pipeline."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.compiled_models = {}
        
    def optimize_model(self, model: nn.Module, model_name: str) -> nn.Module:
        """Apply compute optimizations to a model."""
        
        # Apply gradient checkpointing
        if self.config.enable_gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                logger.info(f"Enabled gradient checkpointing for {model_name}")
        
        # Apply torch.compile optimization
        if self.config.compile_models and hasattr(torch, 'compile'):
            if model_name not in self.compiled_models:
                try:
                    compiled_model = torch.compile(model, mode='reduce-overhead')
                    self.compiled_models[model_name] = compiled_model
                    logger.info(f"Compiled {model_name} with torch.compile")
                    return compiled_model
                except Exception as e:
                    logger.warning(f"Failed to compile {model_name}: {e}")
                    return model
            else:
                return self.compiled_models[model_name]
        
        return model
    
    def optimize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply tensor-level optimizations."""
        
        if self.config.use_channels_last and len(tensor.shape) == 4:
            # Convert to channels_last memory format for better performance
            return tensor.to(memory_format=torch.channels_last)
        
        return tensor


class AsyncProcessor:
    """Asynchronous processing pipeline for overlapping operations."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=2) if config.enable_async_processing else None
        self.frame_queue = queue.Queue(maxsize=10)
        
    def process_async(
        self, 
        processing_func: Callable, 
        input_data: Any, 
        callback: Optional[Callable] = None
    ):
        """Process data asynchronously with optional callback."""
        
        if not self.config.enable_async_processing or self.executor is None:
            # Synchronous processing fallback
            result = processing_func(input_data)
            if callback:
                callback(result)
            return result
        
        # Asynchronous processing
        future = self.executor.submit(processing_func, input_data)
        
        if callback:
            future.add_done_callback(lambda f: callback(f.result()))
        
        return future
    
    def shutdown(self):
        """Shutdown async processing."""
        if self.executor:
            self.executor.shutdown(wait=True)


class FeatureCache:
    """Intelligent caching system for expensive feature computations."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.cache = {}
        self.cache_size_bytes = 0
        self.access_counts = {}
        self.access_order = []
        
    def get(self, key: str) -> Optional[torch.Tensor]:
        """Get cached feature if available."""
        
        if not self.config.enable_feature_caching:
            return None
            
        if key in self.cache:
            # Update access tracking
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            self.access_order.append(key)
            return self.cache[key]
        
        return None
    
    def put(self, key: str, tensor: torch.Tensor):
        """Cache a feature tensor."""
        
        if not self.config.enable_feature_caching:
            return
            
        # Calculate tensor size in bytes
        tensor_size = tensor.element_size() * tensor.numel()
        
        # Check if we need to evict items
        while (self.cache_size_bytes + tensor_size > self.config.cache_size_limit_mb * 1024**2 
               and len(self.cache) > 0):
            self._evict_lru_item()
        
        # Add to cache
        self.cache[key] = tensor.clone()
        self.cache_size_bytes += tensor_size
        self.access_counts[key] = 1
        self.access_order.append(key)
        
    def _evict_lru_item(self):
        """Evict least recently used item from cache."""
        
        if not self.access_order:
            return
        
        # Find least recently used key
        lru_key = self.access_order[0]
        self.access_order.remove(lru_key)
        
        if lru_key in self.cache:
            tensor = self.cache[lru_key]
            tensor_size = tensor.element_size() * tensor.numel()
            
            del self.cache[lru_key]
            del self.access_counts[lru_key]
            self.cache_size_bytes -= tensor_size
    
    def clear(self):
        """Clear all cached items."""
        self.cache.clear()
        self.cache_size_bytes = 0
        self.access_counts.clear()
        self.access_order.clear()


class DualEncoderPerformanceOptimizer:
    """Main performance optimizer for dual-encoder architecture."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        
        # Initialize optimization components
        self.memory_manager = MemoryManager(config)
        self.compute_optimizer = ComputeOptimizer(config)
        self.async_processor = AsyncProcessor(config)
        self.feature_cache = FeatureCache(config)
        
        # Performance tracking
        self.performance_stats = {
            'total_processing_time': 0.0,
            'vggt_processing_time': 0.0,
            'dinov3_processing_time': 0.0,
            'sampling_time': 0.0,
            'fusion_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def optimize_vggt_processing(
        self, 
        vggt_model: nn.Module, 
        video_frames: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Optimize VGGT processing with memory and compute optimizations."""
        
        start_time = time.time()
        
        # Optimize model
        optimized_vggt = self.compute_optimizer.optimize_model(vggt_model, "VGGT")
        
        # Check cache first
        cache_key = self._generate_cache_key("vggt", video_frames)
        cached_result = self.feature_cache.get(cache_key)
        
        if cached_result is not None and self.config.cache_vggt_features:
            self.performance_stats['cache_hits'] += 1
            return cached_result
        
        self.performance_stats['cache_misses'] += 1
        
        # Memory-efficient processing
        with self.memory_manager.memory_efficient_context():
            batch_size, total_frames = video_frames.shape[:2]
            
            # Optimize tensor memory layout
            optimized_frames = self.compute_optimizer.optimize_tensor(video_frames)
            
            # Calculate optimal batch size for frames
            if torch.cuda.is_available():
                available_memory = (torch.cuda.get_device_properties(0).total_memory 
                                  - torch.cuda.memory_allocated()) / 1024**2
            else:
                available_memory = 4096  # Assume 4GB for CPU
            
            frame_memory_mb = (optimized_frames.element_size() * optimized_frames[0, 0].numel()) / 1024**2
            optimal_batch_size = self.memory_manager.get_optimal_batch_size(
                total_frames, frame_memory_mb, available_memory
            )
            
            # Process frames in optimal batches
            all_results = []
            for i in range(0, total_frames, optimal_batch_size):
                end_idx = min(i + optimal_batch_size, total_frames)
                batch_frames = optimized_frames[:, i:end_idx]
                
                # Flatten for VGGT processing
                batch_frames_flat = batch_frames.view(-1, *batch_frames.shape[2:])
                
                # Process with mixed precision if enabled
                with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
                    batch_result = optimized_vggt(batch_frames_flat)
                
                # Reshape back to batch structure
                for key, tensor in batch_result.items():
                    if len(tensor.shape) >= 2:
                        new_shape = (batch_size, end_idx - i) + tensor.shape[1:]
                        batch_result[key] = tensor.view(new_shape)
                
                all_results.append(batch_result)
            
            # Concatenate all batch results
            final_result = {}
            for key in all_results[0].keys():
                concatenated = torch.cat([result[key] for result in all_results], dim=1)
                final_result[key] = concatenated
        
        # Cache results
        if self.config.cache_vggt_features:
            self.feature_cache.put(cache_key, final_result)
        
        # Update performance stats
        processing_time = time.time() - start_time
        self.performance_stats['vggt_processing_time'] += processing_time
        
        return final_result
    
    def optimize_dinov3_processing(
        self, 
        dinov3_model: nn.Module, 
        keyframe_images: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Optimize DINOv3 processing for selected keyframes."""
        
        start_time = time.time()
        
        # Optimize model
        optimized_dinov3 = self.compute_optimizer.optimize_model(dinov3_model, "DINOv3")
        
        # Check cache
        cache_key = self._generate_cache_key("dinov3", keyframe_images)
        cached_result = self.feature_cache.get(cache_key)
        
        if cached_result is not None:
            self.performance_stats['cache_hits'] += 1
            return cached_result
        
        self.performance_stats['cache_misses'] += 1
        
        # Memory-efficient processing
        with self.memory_manager.memory_efficient_context():
            # Optimize tensor layout
            optimized_keyframes = self.compute_optimizer.optimize_tensor(keyframe_images)
            
            # Process with optimal batch size
            batch_size, num_keyframes = optimized_keyframes.shape[:2]
            max_batch = min(num_keyframes, self.config.max_dinov3_batch_size)
            
            batch_results = []
            for i in range(0, num_keyframes, max_batch):
                end_idx = min(i + max_batch, num_keyframes)
                batch_keyframes = optimized_keyframes[:, i:end_idx]
                
                # Process through DINOv3
                with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
                    batch_result = optimized_dinov3(batch_keyframes, return_features=True)
                
                batch_results.append(batch_result)
            
            # Concatenate batch results
            if len(batch_results) == 1:
                final_result = batch_results[0]
            else:
                final_result = {}
                for key in batch_results[0].keys():
                    if isinstance(batch_results[0][key], torch.Tensor):
                        concatenated = torch.cat([result[key] for result in batch_results], dim=1)
                        final_result[key] = concatenated
                    else:
                        final_result[key] = batch_results[0][key]  # Non-tensor items
        
        # Cache results
        self.feature_cache.put(cache_key, final_result)
        
        # Update performance stats
        processing_time = time.time() - start_time
        self.performance_stats['dinov3_processing_time'] += processing_time
        
        return final_result
    
    def optimize_feature_fusion(
        self, 
        vggt_features: torch.Tensor, 
        dinov3_features: torch.Tensor,
        fusion_module: nn.Module
    ) -> torch.Tensor:
        """Optimize feature fusion process."""
        
        start_time = time.time()
        
        # Optimize fusion module
        optimized_fusion = self.compute_optimizer.optimize_model(fusion_module, "FeatureFusion")
        
        # Memory-efficient fusion
        with self.memory_manager.memory_efficient_context():
            with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
                fused_features = optimized_fusion(vggt_features, dinov3_features)
        
        # Update performance stats
        processing_time = time.time() - start_time
        self.performance_stats['fusion_time'] += processing_time
        
        return fused_features
    
    def _generate_cache_key(self, prefix: str, tensor: torch.Tensor) -> str:
        """Generate cache key for tensor-based operations."""
        
        # Create hash based on tensor properties
        tensor_info = (
            prefix,
            tensor.shape,
            tensor.device,
            tensor.dtype,
            hash(tensor.data_ptr())  # Memory address as unique identifier
        )
        
        return f"{prefix}_{hash(tensor_info)}"
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        
        total_time = self.performance_stats['total_processing_time']
        
        report = {
            'performance_stats': self.performance_stats.copy(),
            'cache_statistics': {
                'cache_hits': self.performance_stats['cache_hits'],
                'cache_misses': self.performance_stats['cache_misses'],
                'hit_rate': (self.performance_stats['cache_hits'] / 
                           max(self.performance_stats['cache_hits'] + self.performance_stats['cache_misses'], 1)),
                'cache_size_mb': self.feature_cache.cache_size_bytes / 1024**2,
                'cached_items': len(self.feature_cache.cache)
            },
            'memory_statistics': self.memory_manager.memory_stats[-10:] if self.memory_manager.memory_stats else [],
            'optimization_config': self.config.__dict__
        }
        
        if total_time > 0:
            report['time_breakdown'] = {
                'vggt_percentage': (self.performance_stats['vggt_processing_time'] / total_time) * 100,
                'dinov3_percentage': (self.performance_stats['dinov3_processing_time'] / total_time) * 100,
                'sampling_percentage': (self.performance_stats['sampling_time'] / total_time) * 100,
                'fusion_percentage': (self.performance_stats['fusion_time'] / total_time) * 100
            }
        
        return report
    
    def cleanup(self):
        """Cleanup resources and shutdown async processing."""
        self.async_processor.shutdown()
        self.feature_cache.clear()


def create_performance_optimizer(
    enable_all_optimizations: bool = True,
    max_memory_mb: int = 12000,
    target_fps: float = 25.0
) -> DualEncoderPerformanceOptimizer:
    """
    Factory function to create performance optimizer.
    
    Args:
        enable_all_optimizations: Enable all available optimizations
        max_memory_mb: Maximum memory usage limit
        target_fps: Target processing speed
        
    Returns:
        Configured performance optimizer
    """
    
    config = PerformanceConfig(
        enable_gradient_checkpointing=enable_all_optimizations,
        use_mixed_precision=enable_all_optimizations,
        compile_models=enable_all_optimizations and hasattr(torch, 'compile'),
        enable_async_processing=enable_all_optimizations,
        cache_size_limit_mb=max_memory_mb // 4,  # Use 1/4 of memory for cache
        adaptive_batch_size=True,
        log_optimization_stats=True
    )
    
    return DualEncoderPerformanceOptimizer(config)


# Example usage and benchmarking
if __name__ == "__main__":
    optimizer = create_performance_optimizer()
    
    print("Dual-Encoder Performance Optimizer created successfully!")
    print(f"Configuration: {optimizer.config}")
    
    # Cleanup
    optimizer.cleanup()