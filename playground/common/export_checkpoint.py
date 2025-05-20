"""
Script to export Orbax checkpoints to ONNX format.
Usage: python -m playground.common.export_checkpoint --checkpoint /path/to/checkpoint --output exported_checkpoints/model.onnx
"""

import argparse
import os
import tensorflow as tf
from tensorflow.keras import layers
import tf2onnx
import numpy as np
import orbax.checkpoint as ocp
import jax

def create_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def export_checkpoint(checkpoint_path, output_path, obs_size=101, act_size=14):
    """Export a checkpoint to ONNX format."""
    print(f"Exporting checkpoint: {checkpoint_path}")
    print(f"Output path: {output_path}")
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:
        create_directory(output_dir)
    
    # Load the checkpoint
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    params = orbax_checkpointer.restore(checkpoint_path)
    
    # Print parameter structure for debugging
    print("\nParameter structure:")
    print(f"Type: {type(params)}")
    if isinstance(params, (list, tuple)):
        print(f"Length: {len(params)}")
        for i, p in enumerate(params):
            print(f"  params[{i}] type: {type(p)}")
            if hasattr(p, 'keys'):
                print(f"  params[{i}] keys: {list(p.keys())}")
    
    # Create a simple MLP policy network
    inputs = layers.Input(shape=(obs_size,), name="obs")
    
    # Create MLP layers
    with tf.name_scope("MLP_0"):
        x = layers.Dense(256, activation="tanh", name="hidden_0")(inputs)
        x = layers.Dense(256, activation="tanh", name="hidden_1")(x)
        x = layers.Dense(act_size, activation=None, name="hidden_2")(x)
    
    tf_policy_network = tf.keras.Model(inputs=inputs, outputs=x)
    
    # Define the TensorFlow input signature
    spec = [
        tf.TensorSpec(shape=(1, obs_size), dtype=tf.float32, name="obs")
    ]
    
    # Set output names
    tf_policy_network.output_names = ["continuous_actions"]
    
    # Convert to ONNX
    model_proto, _ = tf2onnx.convert.from_keras(
        tf_policy_network, input_signature=spec, opset=11, output_path=output_path
    )
    
    print(f"\nExported model to {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Export checkpoint to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--output", type=str, default="exported_checkpoints/model.onnx", help="Output ONNX path")
    parser.add_argument("--obs_size", type=int, default=101, help="Observation size")
    parser.add_argument("--act_size", type=int, default=14, help="Action size")
    args = parser.parse_args()
    
    # Convert to absolute path if needed
    checkpoint_path = os.path.abspath(args.checkpoint)
    
    # Export the checkpoint
    export_checkpoint(
        checkpoint_path=checkpoint_path,
        output_path=args.output,
        obs_size=args.obs_size,
        act_size=args.act_size
    )

if __name__ == "__main__":
    main() 