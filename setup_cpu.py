"""Setup script for CPU-only installation"""
from setuptools import setup

setup(
    name="Open_Duck_Playground",
    version="0.1.0",
    description="Open Duck Playground (CPU version)",
    packages=["playground"],
    install_requires=[
        "framesviewer>=1.0.2",
        "jax>=0.5.0",  # CPU-only
        "jaxlib>=0.5.0",
        "jaxtyping>=0.2.38",
        "matplotlib>=3.10.0",
        "mediapy>=1.2.2",
        "onnxruntime>=1.20.1",
        "playground>=0.0.3",
        "pygame>=2.6.1",
        "tensorflow>=2.18.0",
        "tf2onnx>=1.16.1",
    ],
    python_requires=">=3.11",
) 