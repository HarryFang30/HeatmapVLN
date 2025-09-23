#!/usr/bin/env python3
"""
VLN Pipeline Test Script
========================

Tests the complete VLN pipeline with three inputs:
1. Current observation (first-person view)
2. Feature tokens from VGGT and DINOv3 (spatial understanding)
3. Language instructions (navigation commands)

Outputs a single first-person inter-frame heatmap showing spatial relationships
between video frames from the current observation's perspective.
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse
import json

# Add project root to path (go up two levels from tests/scripts/ to Project/)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.spatial_mllm_compat import create_spatial_mllm_pipeline


def load_video_frames(video_path: str, max_frames: int = 16, target_size: tuple = (224, 224)) -> torch.Tensor:
    """Load video frames from file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video has {total_frames} frames, loading {max_frames}")
    
    # Sample frames evenly
    if total_frames > max_frames:
        step = total_frames // max_frames
        frame_indices = list(range(0, total_frames, step))[:max_frames]
    else:
        frame_indices = list(range(total_frames))
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
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
    
    return torch.stack(frames[:max_frames])  # [max_frames, 3, H, W]


def load_instruction_from_file(instruction_path: str) -> str:
    """Load language instruction from file."""
    if instruction_path.endswith('.json'):
        with open(instruction_path, 'r') as f:
            data = json.load(f)
            # Assume JSON has 'instruction' field
            return data.get('instruction', data.get('text', str(data)))
    else:
        # Plain text file
        with open(instruction_path, 'r') as f:
            return f.read().strip()


def test_pipeline(video_path: str, instruction: str, output_dir: str = None):
    """Test the complete VLN pipeline with three inputs."""
    print("=== VLN Pipeline Test ===")
    print("Testing three-input processing:")
    print("  1. Current observation (first-person view)")
    print("  2. Feature tokens from video (spatial understanding)")
    print("  3. Language instruction (navigation command)")
    print()
    
    # Set default output directory relative to tests/
    if output_dir is None:
        output_dir = str(Path(__file__).parent.parent / "outputs" / "test_results")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load test video
    print(f"Loading video: {video_path}")
    
    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        return False
    
    video_frames = load_video_frames(video_path, max_frames=8, target_size=(224, 224))
    print(f"Loaded video frames: {video_frames.shape}")
    
    # Add batch dimension
    video_frames = video_frames.unsqueeze(0)  # [1, N_m, C, H, W]
    print(f"Batch video frames: {video_frames.shape}")
    
    # Use middle frame as current observation (more representative than first frame)
    middle_idx = video_frames.shape[1] // 2
    current_observation = video_frames[:, middle_idx]  # [1, C, H, W]
    print(f"Current observation shape: {current_observation.shape}")
    print(f"Using middle frame (index {middle_idx} of {video_frames.shape[1]}) as current observation")
    
    print(f"Language instruction: '{instruction}'")
    print()
    
    # Create pipeline
    print("Creating VLN pipeline...")
    pipeline = create_spatial_mllm_pipeline(
        target_keyframes=4,
        total_frames=8,
        dinov3_model_size="base",  # Use smaller model
        img_size=224,  # Use smaller image size to fit in memory
        device=device,
        verbose=True  # Enable diagnostic messages for transparency
    )
    print("Pipeline created successfully!")
    
    # Move tensors to device
    video_frames = video_frames.to(device)
    current_observation = current_observation.to(device)
    
    # Run inference with three inputs
    print("Running inference with three inputs...")
    try:
        with torch.no_grad():
            result = pipeline(
                video_frames=video_frames,
                instruction_text=instruction,
                current_observation=current_observation,
                return_intermediate=True,
                return_heatmaps=True
            )
        
        print("Inference completed successfully!")
        
        # Print results
        print(f"\nResults:")
        print(f"  Selected keyframes: {result['selected_keyframes']}")
        print(f"  Fused features shape: {result['fused_features'].shape}")
        print(f"  LLM tokens shape: {result['llm_tokens'].shape}")
        
        # Check for single inter-frame heatmap
        if 'inter_frame_heatmap' in result:
            heatmap = result['inter_frame_heatmap']
            print(f"  Single inter-frame heatmap shape: {heatmap.shape}")
            
            # Visualize the single heatmap
            heatmap_np = heatmap[0, 0].cpu().numpy()  # [H, W]
            
            # Create visualization with current observation and heatmap
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Plot current observation
            current_obs_np = current_observation[0].permute(1, 2, 0).cpu().numpy()
            current_obs_np = np.clip(current_obs_np, 0, 1)
            axes[0].imshow(current_obs_np)
            axes[0].set_title('Current Observation\n(First-Person View)')
            axes[0].axis('off')
            
            # Plot inter-frame heatmap
            im1 = axes[1].imshow(heatmap_np, cmap='viridis')
            axes[1].set_title('Inter-Frame Spatial Heatmap\n(Shows where other frames appear)')
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1])
            
            # Plot overlay
            axes[2].imshow(current_obs_np)
            axes[2].imshow(heatmap_np, alpha=0.6, cmap='hot')
            axes[2].set_title('Overlay: Observation + Heatmap\n(Spatial relationships)')
            axes[2].axis('off')
            
            plt.suptitle(f'VLN Pipeline Results\nInstruction: "{instruction}"', fontsize=14)
            plt.tight_layout()
            
            output_path = Path(output_dir) / 'vln_single_heatmap_results.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  Saved visualization to: {output_path}")
            plt.close()
            
        elif 'inter_frame_heatmaps' in result:
            # Fallback to old format
            heatmaps = result['inter_frame_heatmaps']
            print(f"  Multiple heatmaps shape: {heatmaps.shape}")
        else:
            print("  No heatmaps generated")
        
        print(f"\nProcessing metadata: {result['processing_metadata']}")
        
        # Save metadata
        metadata_path = Path(output_dir) / 'processing_metadata.json'
        with open(metadata_path, 'w') as f:
            # Convert tensors to lists for JSON serialization
            metadata = {}
            for k, v in result['processing_metadata'].items():
                if hasattr(v, 'tolist'):
                    metadata[k] = v.tolist()
                elif hasattr(v, 'shape'):
                    metadata[k] = str(v.shape)
                else:
                    metadata[k] = v
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to: {metadata_path}")
        
        return True
        
    except Exception as e:
        print(f"Inference failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Test VLN pipeline with video and language instruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test with video and text instruction
    python test_pipeline.py --video /path/to/video.mp4 --instruction "Go to the kitchen"
    
    # Test with video and instruction from file
    python test_pipeline.py --video /path/to/video.mp4 --instruction_file /path/to/instruction.txt
    
    # Test with custom output directory
    python test_pipeline.py --video /path/to/video.mp4 --instruction "Turn left and find the door" --output_dir ./my_results
        """
    )
    
    parser.add_argument(
        "--video", 
        type=str, 
        default="./test.mp4",
        help="Path to input video file"
    )
    
    parser.add_argument(
        "--instruction", 
        type=str, 
        help="Language instruction for navigation"
    )
    
    parser.add_argument(
        "--instruction_file",
        type=str,
        help="Path to file containing language instruction (alternative to --instruction)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: tests/outputs/test_results/)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Determine instruction
    if args.instruction:
        instruction = args.instruction
    elif args.instruction_file:
        if not Path(args.instruction_file).exists():
            print(f"Error: Instruction file not found: {args.instruction_file}")
            sys.exit(1)
        instruction = load_instruction_from_file(args.instruction_file)
    else:
        # Default instruction
        instruction = "Navigate through this space and understand the spatial relationships between frames"
    
    print(f"Video: {args.video}")
    print(f"Instruction: {instruction}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    success = test_pipeline(
        video_path=args.video,
        instruction=instruction,
        output_dir=args.output_dir
    )
    
    if success:
        print("\n=== TEST PASSED ===")
        print("VLN pipeline is working correctly!")
        print("✓ Successfully processed three inputs:")
        print("  - Current observation (first-person view)")
        print("  - Feature tokens from video (spatial understanding)")
        print("  - Language instruction (navigation command)")
        print("✓ Generated single first-person inter-frame heatmap")
        print(f"✓ Results saved to: {args.output_dir}")
    else:
        print("\n=== TEST FAILED ===")
        print("VLN pipeline encountered errors.")
        sys.exit(1)