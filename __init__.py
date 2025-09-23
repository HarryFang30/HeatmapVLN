"""
VLN Project - Vision-Language Navigation with First-Person Inter-Frame Heatmap Generation

This project implements a sophisticated VLN pipeline that generates first-person-view heatmaps
showing spatial relationships between video frames, demonstrating the model's understanding
of inter-frame spatial connections.

Architecture:
1. VGGT (3D encoder) processes all N_m frames for geometry extraction
2. Space-aware sampling selects N_k informative keyframes using geometry
3. Dual-path processing: 3D features from VGGT + 2D features from DINOv3
4. MLP fusion creates mixed spatial-semantic features
5. LLM processes features + instructions for spatial reasoning
6. Heatmap generation with cross-frame spatial projection
"""

__version__ = "0.1.0"
__author__ = "VLN Research Team"
__description__ = "First-Person Inter-Frame Spatial Understanding for VLN"