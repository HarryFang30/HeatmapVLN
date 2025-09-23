# VLN Pipeline Tests
=====================

This directory contains comprehensive tests for the VLN (Vision-Language Navigation) pipeline that generates first-person inter-frame heatmaps.

## Directory Structure

```
tests/
├── README.md                 # This file
├── scripts/                  # Test scripts
│   ├── test_pipeline.py      # Main test script  
│   └── run_all_tests.py      # Comprehensive test runner
├── configs/                  # Test configurations
│   └── test_config.yaml      # Main test configuration
├── instructions/             # Test navigation instructions
│   ├── navigation_basic.txt
│   ├── exploration_detailed.txt
│   └── robot_perspective.json
├── data/                     # Test data (uses auto-discovered video locations)
├── outputs/                  # Default test output directory
│   └── test_results/         # Individual test results
└── results/                  # Comprehensive test session results
    └── test_session_*/       # Timestamped test sessions
```

## Quick Start

### Run Single Test
```bash
cd <project_root>

# Basic test with official video
python tests/scripts/test_pipeline.py --video test.mp4 --instruction "Navigate and analyze spatial relationships"

# Test with instruction file
python tests/scripts/test_pipeline.py --video test.mp4 --instruction_file tests/instructions/navigation_basic.txt

# Custom output directory
python tests/scripts/test_pipeline.py --video test.mp4 --instruction "Explore this space" --output_dir tests/outputs/custom_test
```

### Run Comprehensive Test Suite
```bash
cd <project_root>

# Run all tests with default configuration
python tests/scripts/run_all_tests.py

# Run with custom configuration
python tests/scripts/run_all_tests.py --config tests/configs/test_config.yaml
```

## Test Videos Used

The tests automatically discover videos in the following search locations:
- `test.mp4` - Official test video (192 frames)
- `vggt/examples/videos/kitchen.mp4` - Kitchen scene (25 frames)  
- `Spatial-MLLM/assets/arkitscenes_41069025.mp4` - ARKit scene (5045 frames)
- `vggt/examples/videos/single_cartoon.mp4` - Cartoon scene (25 frames)

## Test Instructions

Three types of navigation instructions are tested:
1. **Basic Navigation** - Simple spatial understanding task
2. **Detailed Exploration** - Comprehensive spatial analysis 
3. **Robot Perspective** - Task-oriented navigation scenario

## What Gets Tested

The tests verify:
- ✅ **Three-Input Processing**: Current observation + feature tokens + instructions
- ✅ **Middle Frame Selection**: Uses middle frame as current observation 
- ✅ **Space-Aware Keyframe Selection**: VGGT-based intelligent frame sampling
- ✅ **Dual-Path Feature Extraction**: VGGT (3D) + DINOv3 (2D) features
- ✅ **Feature Fusion**: Spatial-semantic feature combination
- ✅ **Single Heatmap Generation**: First-person inter-frame spatial relationships
- ✅ **Output Organization**: Proper file structure and metadata

## Expected Outputs

Each test produces:
- `vln_single_heatmap_results.png` - Visualization with 3 panels:
  - Current observation (middle frame)
  - Inter-frame spatial heatmap
  - Overlay showing spatial relationships
- `processing_metadata.json` - Detailed processing statistics

## Comprehensive Test Results

Running `run_all_tests.py` creates a timestamped session directory with:
- Individual test results for each video/instruction combination
- Overall test summary and statistics
- Detailed JSON results file
- Success/failure analysis

## Configuration

Modify `tests/configs/test_config.yaml` to:
- Change model parameters (DINOv3 size, keyframe count, etc.)
- Add new test videos or instructions
- Adjust output settings
- Configure timeout and validation settings

## Troubleshooting

**Path Issues**: All paths are configured to work with automatic path resolution from the project root.

**GPU Memory**: Tests use smaller models (base size) to fit in memory. Increase model size in config if you have more VRAM.

**Missing Videos**: Ensure test videos exist in discoverable locations. The system searches multiple common locations automatically.

**Permission Errors**: Test scripts are made executable with `chmod +x tests/scripts/*.py`.