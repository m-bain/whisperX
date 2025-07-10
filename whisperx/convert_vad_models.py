#!/usr/bin/env python3
"""
Weight conversion tools for Silero VAD models
Converts ONNX models to MLX format
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import onnx
import onnxruntime as ort


class SileroVADMLX(nn.Module):
    """MLX implementation of Silero VAD model."""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Model configuration
        self.sample_rate = config.get("sample_rate", 16000)
        self.window_size_samples = config.get("window_size_samples", 512)
        
        # Build model layers based on ONNX structure
        # Silero VAD uses LSTM layers
        self.lstm1 = nn.LSTM(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            batch_first=True
        )
        
        self.lstm2 = nn.LSTM(
            input_size=config["hidden_size"],
            hidden_size=config["hidden_size"],
            batch_first=True
        )
        
        # Output projection
        self.output = nn.Linear(config["hidden_size"], config["num_classes"])
        
        # Activation
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: mx.array) -> mx.array:
        """Forward pass."""
        # LSTM layers
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        
        # Output projection
        x = self.output(x)
        x = self.sigmoid(x)
        
        return x


def extract_onnx_weights(onnx_path: str) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Extract weights from ONNX model.
    
    Args:
        onnx_path: Path to ONNX model
        
    Returns:
        Weights dictionary and model config
    """
    print(f"Loading ONNX model from {onnx_path}...")
    
    # Load ONNX model
    model = onnx.load(onnx_path)
    
    # Create ONNX runtime session to get weights
    session = ort.InferenceSession(onnx_path)
    
    # Extract model structure
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    
    # Get input/output shapes
    input_shape = input_info.shape
    output_shape = output_info.shape
    
    print(f"Input shape: {input_shape}")
    print(f"Output shape: {output_shape}")
    
    # Extract weights from initializers
    weights = {}
    
    for initializer in model.graph.initializer:
        name = initializer.name
        # Convert to numpy array
        weight = onnx.numpy_helper.to_array(initializer)
        weights[name] = weight
        print(f"  Found weight: {name} {weight.shape}")
    
    # Infer model configuration
    # Silero VAD typically has specific architecture
    config = {
        "sample_rate": 16000,
        "window_size_samples": 512,
        "input_size": 1,  # Single channel audio
        "hidden_size": 64,  # LSTM hidden size
        "num_classes": 1,  # Binary classification (speech/non-speech)
    }
    
    # Try to infer sizes from weights
    for name, weight in weights.items():
        if "lstm" in name.lower() and "weight_ih" in name:
            # Input-to-hidden weight shape: (4*hidden_size, input_size)
            config["hidden_size"] = weight.shape[0] // 4
            config["input_size"] = weight.shape[1]
            break
    
    return weights, config


def map_silero_weights(onnx_weights: Dict[str, np.ndarray], config: Dict) -> Dict[str, mx.array]:
    """Map ONNX weights to MLX format.
    
    Args:
        onnx_weights: ONNX weight dictionary
        config: Model configuration
        
    Returns:
        MLX weight dictionary
    """
    mlx_weights = {}
    
    # Mapping patterns for Silero VAD
    # ONNX uses different naming conventions
    mappings = {
        # LSTM layers - ONNX packs all LSTM weights together
        # Format: weight_ih (input-to-hidden), weight_hh (hidden-to-hidden)
        # Each contains weights for: input, forget, cell, output gates
        "lstm.weight_ih_l0": "lstm1.weight_ih",
        "lstm.weight_hh_l0": "lstm1.weight_hh",
        "lstm.bias_ih_l0": "lstm1.bias_ih",
        "lstm.bias_hh_l0": "lstm1.bias_hh",
        
        "lstm.weight_ih_l1": "lstm2.weight_ih",
        "lstm.weight_hh_l1": "lstm2.weight_hh",
        "lstm.bias_ih_l1": "lstm2.bias_ih",
        "lstm.bias_hh_l1": "lstm2.bias_hh",
        
        # Output layer
        "fc.weight": "output.weight",
        "fc.bias": "output.bias",
    }
    
    # Try alternative naming schemes
    alternative_mappings = {
        # Some models use numbered naming
        "0.weight_ih_l0": "lstm1.weight_ih",
        "0.weight_hh_l0": "lstm1.weight_hh",
        "0.bias_ih_l0": "lstm1.bias_ih",
        "0.bias_hh_l0": "lstm1.bias_hh",
        
        "1.weight_ih_l0": "lstm2.weight_ih",
        "1.weight_hh_l0": "lstm2.weight_hh", 
        "1.bias_ih_l0": "lstm2.bias_ih",
        "1.bias_hh_l0": "lstm2.bias_hh",
        
        "2.weight": "output.weight",
        "2.bias": "output.bias",
    }
    
    # Apply mappings
    for onnx_name, weight in onnx_weights.items():
        mlx_name = None
        
        # Try primary mappings
        for pattern, target in mappings.items():
            if pattern in onnx_name or onnx_name == pattern:
                mlx_name = target
                break
        
        # Try alternative mappings if not found
        if not mlx_name:
            for pattern, target in alternative_mappings.items():
                if pattern in onnx_name or onnx_name == pattern:
                    mlx_name = target
                    break
        
        if mlx_name:
            # Handle weight transformations
            if "lstm" in mlx_name and "weight" in mlx_name:
                # LSTM weights may need reshaping
                # ONNX format: (num_directions * 4 * hidden_size, input_size)
                # MLX expects same format
                pass
            elif "output.weight" in mlx_name and weight.ndim == 2:
                # Linear layer weights may need transposing
                weight = weight.T
            
            mlx_weights[mlx_name] = mx.array(weight)
            print(f"  {onnx_name} -> {mlx_name} {weight.shape}")
        else:
            print(f"  Warning: No mapping for {onnx_name}")
    
    return mlx_weights


def convert_silero_vad(
    onnx_path: str = "silero_vad.onnx",
    output_path: str = "mlx_models/silero_vad_mlx",
    test_conversion: bool = True
) -> bool:
    """Convert Silero VAD model from ONNX to MLX.
    
    Args:
        onnx_path: Path to ONNX model
        output_path: Output directory for MLX model
        test_conversion: Whether to test the conversion
        
    Returns:
        True if successful
    """
    print(f"Converting Silero VAD to MLX format...")
    
    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Extract ONNX weights
        onnx_weights, config = extract_onnx_weights(onnx_path)
        
        # Save config
        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # Convert weights to MLX
        print("\nConverting weights...")
        mlx_weights = map_silero_weights(onnx_weights, config)
        
        # Save weights
        print(f"Saving to {output_dir / 'weights.npz'}...")
        mx.save(str(output_dir / "weights.npz"), mlx_weights)
        
        print(f"✓ Model converted successfully!")
        
        # Test conversion if requested
        if test_conversion:
            print("\nTesting conversion...")
            
            # Create MLX model
            model = SileroVADMLX(config)
            
            # Load weights
            model.load_weights(list(mlx_weights.items()))
            
            # Test forward pass
            test_input = mx.random.normal((1, 100, config["input_size"]))
            output = model(test_input)
            print(f"✓ Test forward pass successful! Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_silero_vad() -> str:
    """Download Silero VAD ONNX model.
    
    Returns:
        Path to downloaded model
    """
    import urllib.request
    
    url = "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx"
    output_path = "silero_vad.onnx"
    
    if not Path(output_path).exists():
        print(f"Downloading Silero VAD from {url}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"✓ Downloaded to {output_path}")
    
    return output_path


def convert_all_vad_models(output_dir: str = "~/mlx_models/vad"):
    """Convert all standard VAD models."""
    output_dir = Path(output_dir).expanduser()
    
    models = [
        ("silero_vad.onnx", "silero_vad_mlx"),
        # Add other VAD models here if needed
    ]
    
    results = []
    
    for onnx_model, mlx_name in models:
        print(f"\n{'='*60}")
        print(f"Converting {onnx_model}")
        print('='*60)
        
        # Download if needed
        if onnx_model == "silero_vad.onnx" and not Path(onnx_model).exists():
            onnx_model = download_silero_vad()
        
        success = convert_silero_vad(
            onnx_model,
            str(output_dir / mlx_name),
            test_conversion=True
        )
        
        results.append((onnx_model, success))
    
    # Print summary
    print("\n" + "="*60)
    print("CONVERSION SUMMARY")
    print("="*60)
    
    for model, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {model}")
    
    print(f"\nModels saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Silero VAD models to MLX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single model
  python convert_vad_models.py --onnx silero_vad.onnx --output ~/mlx_models/silero_vad_mlx

  # Download and convert Silero VAD
  python convert_vad_models.py --download --output ~/mlx_models/silero_vad_mlx

  # Convert all standard models
  python convert_vad_models.py --all --output-dir ~/mlx_models/vad

  # Test existing conversion
  python convert_vad_models.py --test ~/mlx_models/silero_vad_mlx
""")
    
    parser.add_argument("--onnx", 
                       help="ONNX model path")
    parser.add_argument("--output",
                       help="Output path for MLX model")
    parser.add_argument("--download", action="store_true",
                       help="Download Silero VAD first")
    parser.add_argument("--all", action="store_true",
                       help="Convert all standard models")
    parser.add_argument("--output-dir", default="~/mlx_models/vad",
                       help="Output directory for --all")
    parser.add_argument("--test",
                       help="Test existing MLX model")
    
    args = parser.parse_args()
    
    if args.test:
        # Test existing model
        print(f"Testing {args.test}...")
        
        # Load config
        config_path = Path(args.test) / "config.json"
        with open(config_path) as f:
            config = json.load(f)
        
        # Create model
        model = SileroVADMLX(config)
        
        # Load weights
        weights = mx.load(str(Path(args.test) / "weights.npz"))
        model.load_weights(list(weights.items()))
        
        # Test
        test_input = mx.random.normal((1, 100, config["input_size"]))
        output = model(test_input)
        print(f"✓ Model loaded successfully! Output shape: {output.shape}")
        
    elif args.all:
        convert_all_vad_models(args.output_dir)
    elif args.download and args.output:
        onnx_path = download_silero_vad()
        success = convert_silero_vad(onnx_path, args.output)
        if not success:
            exit(1)
    elif args.onnx and args.output:
        success = convert_silero_vad(args.onnx, args.output)
        if not success:
            exit(1)
    else:
        parser.error("Specify --onnx and --output, or use --download, --all, or --test")


if __name__ == "__main__":
    main()