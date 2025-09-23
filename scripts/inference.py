"""
Inference script for VLN Spatial-MLLM Pipeline
Processes video sequences and generates first-person inter-frame heatmaps
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger


def load_video_frames(video_path: str, max_frames: int = 32, target_size: tuple = (224, 224)) -> torch.Tensor:
    """
    Load video frames from file.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to load
        target_size: Target frame size (H, W)
    
    Returns:
        Video frames tensor of shape [max_frames, 3, H, W]
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame
        frame = cv2.resize(frame, target_size)
        
        # Convert to tensor and normalize
        frame = torch.from_numpy(frame).float() / 255.0
        frame = frame.permute(2, 0, 1)  # HWC to CHW
        
        frames.append(frame)
    
    cap.release()
    
    # Pad if necessary
    while len(frames) < max_frames:
        frames.append(torch.zeros_like(frames[0]))
    
    return torch.stack(frames)  # [max_frames, 3, H, W]


def visualize_heatmaps(heatmaps: torch.Tensor, output_dir: str, video_name: str):
    """
    Visualize and save heatmaps.
    
    Args:
        heatmaps: Tensor of shape [num_views, H, W] or [B, num_views, H, W]
        output_dir: Output directory
        video_name: Name for saving files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if heatmaps.dim() == 4:  # Batch dimension
        heatmaps = heatmaps[0]  # Take first batch
    
    num_views = heatmaps.shape[0]
    
    # Create subplot grid
    cols = min(num_views, 4)
    rows = (num_views + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(num_views):
        row, col = i // cols, i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        heatmap = heatmaps[i].cpu().numpy()
        
        im = ax.imshow(heatmap, cmap='viridis', interpolation='bilinear')
        ax.set_title(f'Inter-Frame Heatmap {i+1}')
        ax.axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for i in range(num_views, rows * cols):
        row, col = i // cols, i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{video_name}_heatmaps.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save individual heatmaps
    for i in range(num_views):
        heatmap = heatmaps[i].cpu().numpy()
        
        # Normalize to 0-255
        heatmap_normalized = ((heatmap - heatmap.min()) / 
                             (heatmap.max() - heatmap.min() + 1e-8) * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap_colored = plt.cm.viridis(heatmap_normalized / 255.0)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Save as image
        Image.fromarray(heatmap_colored).save(
            os.path.join(output_dir, f"{video_name}_heatmap_{i+1}.png")
        )


def save_inference_results(results: Dict[str, Any], output_dir: str, video_name: str):
    """Save inference results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save selected indices
    if 'selected_indices' in results:
        np.save(
            os.path.join(output_dir, f"{video_name}_selected_indices.npy"),
            results['selected_indices']
        )
    
    # Save geometry info
    if 'geometry_info' in results:
        geometry_data = {}
        for key, value in results['geometry_info'].items():
            if isinstance(value, torch.Tensor):
                geometry_data[key] = value.cpu().numpy()
            else:
                geometry_data[key] = value
        
        np.savez(
            os.path.join(output_dir, f"{video_name}_geometry_info.npz"),
            **geometry_data
        )
    
    # Save results summary
    summary = {
        'video_name': video_name,
        'processing_time': results.get('processing_time', 0),
        'num_selected_frames': len(results.get('selected_indices', [])),
        'heatmap_shape': list(results['first_person_heatmaps'].shape) if 'first_person_heatmaps' in results else None,
        'success': results.get('success', False)
    }
    
    import json
    with open(os.path.join(output_dir, f"{video_name}_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)


def inference_pipeline(pipeline, config: Dict[str, Any], args):
    """
    Main inference function called from main.py
    
    Args:
        pipeline: VLN pipeline instance  
        config: Configuration dictionary
        args: Command line arguments
    """
    logger = setup_logger("inference", config['logging']['level'])
    logger.info("Starting VLN inference pipeline")
    
    video_path = args.video_path
    instruction = args.instruction or "Navigate and understand spatial relationships between frames"
    
    # Validate inputs
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    video_name = Path(video_path).stem
    output_dir = args.output_dir or os.path.join(config['paths']['output_dir'], 'inference')
    
    logger.info(f"Processing video: {video_path}")
    logger.info(f"Instruction: {instruction}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load video frames
    logger.info("Loading video frames...")
    start_time = time.time()
    
    video_frames = load_video_frames(
        video_path, 
        max_frames=config['video']['total_frames'],
        target_size=tuple(config['video']['frame_size'])
    )
    
    # Add batch dimension
    video_frames = video_frames.unsqueeze(0)  # [1, N_m, 3, H, W]
    text_instructions = [instruction]
    
    load_time = time.time() - start_time
    logger.info(f"Video loaded: {video_frames.shape} frames in {load_time:.2f}s")
    
    # Set pipeline to evaluation mode
    pipeline.eval_mode()
    
    # Run inference
    logger.info("Running inference...")
    inference_start = time.time()
    
    with torch.no_grad():
        try:
            results = pipeline.forward(
                video_frames=video_frames,
                text_instructions=text_instructions,
                current_view_frame=video_frames[:, 0]  # Use first frame as current view
            )
            
            inference_time = time.time() - inference_start
            results['processing_time'] = inference_time
            results['success'] = True
            
            logger.info(f"Inference completed in {inference_time:.2f}s")
            logger.info(f"Generated heatmaps: {results['first_person_heatmaps'].shape}")
            logger.info(f"Selected {len(results['selected_indices'][0])} keyframes from {video_frames.shape[1]} total frames")
            
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            results = {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - inference_start
            }
    
    if results.get('success', False):
        # Visualize results
        logger.info("Generating visualizations...")
        visualize_heatmaps(
            results['first_person_heatmaps'],
            output_dir,
            video_name
        )
        
        # Save results
        logger.info("Saving results...")
        save_inference_results(results, output_dir, video_name)
        
        logger.info(f"Results saved to: {output_dir}")
        
        # Print summary
        print("\n" + "="*50)
        print("INFERENCE SUMMARY")
        print("="*50)
        print(f"Video: {video_name}")
        print(f"Instruction: {instruction}")
        print(f"Processing time: {results['processing_time']:.2f}s")
        print(f"Selected keyframes: {len(results['selected_indices'][0])}/{video_frames.shape[1]}")
        print(f"Generated heatmaps: {results['first_person_heatmaps'].shape}")
        print(f"Output directory: {output_dir}")
        print("="*50)
        
    else:
        logger.error("Inference failed - no results to save")
        print(f"Inference failed: {results.get('error', 'Unknown error')}")
    
    return results


if __name__ == "__main__":
    print("VLN Inference Script - Run via main.py for full functionality")