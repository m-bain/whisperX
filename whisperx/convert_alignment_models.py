#!/usr/bin/env python3
"""
Weight conversion tools for Wav2Vec2 alignment models
Converts HuggingFace models to MLX format
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def map_wav2vec2_weights(state_dict: Dict[str, torch.Tensor]) -> Dict[str, mx.array]:
    """Map PyTorch Wav2Vec2 weights to MLX format.
    
    Args:
        state_dict: PyTorch model state dict
        
    Returns:
        Dictionary of MLX arrays
    """
    mlx_weights = {}
    
    # Mapping rules for different layer types
    mappings = {
        # Feature extractor CNN layers
        "wav2vec2.feature_extractor.conv_layers.{}.conv.weight": "feature_extractor.conv{}.weight",
        "wav2vec2.feature_extractor.conv_layers.{}.conv.bias": "feature_extractor.conv{}.bias",
        "wav2vec2.feature_extractor.conv_layers.{}.layer_norm.weight": "feature_extractor.norm{}.weight",
        "wav2vec2.feature_extractor.conv_layers.{}.layer_norm.bias": "feature_extractor.norm{}.bias",
        
        # Feature projection
        "wav2vec2.feature_projection.projection.weight": "feature_projection.weight",
        "wav2vec2.feature_projection.projection.bias": "feature_projection.bias",
        "wav2vec2.feature_projection.layer_norm.weight": "layer_norm.weight",
        "wav2vec2.feature_projection.layer_norm.bias": "layer_norm.bias",
        
        # Positional conv embedding
        "wav2vec2.encoder.pos_conv_embed.conv.weight_g": "pos_conv_embed.conv.weight_g",
        "wav2vec2.encoder.pos_conv_embed.conv.weight_v": "pos_conv_embed.conv.weight_v",
        "wav2vec2.encoder.pos_conv_embed.conv.bias": "pos_conv_embed.conv.bias",
        
        # Encoder layers
        "wav2vec2.encoder.layers.{}.attention.q_proj.weight": "encoder_layers.{}.attention.q_proj.weight",
        "wav2vec2.encoder.layers.{}.attention.q_proj.bias": "encoder_layers.{}.attention.q_proj.bias",
        "wav2vec2.encoder.layers.{}.attention.k_proj.weight": "encoder_layers.{}.attention.k_proj.weight",
        "wav2vec2.encoder.layers.{}.attention.k_proj.bias": "encoder_layers.{}.attention.k_proj.bias",
        "wav2vec2.encoder.layers.{}.attention.v_proj.weight": "encoder_layers.{}.attention.v_proj.weight",
        "wav2vec2.encoder.layers.{}.attention.v_proj.bias": "encoder_layers.{}.attention.v_proj.bias",
        "wav2vec2.encoder.layers.{}.attention.out_proj.weight": "encoder_layers.{}.attention.out_proj.weight",
        "wav2vec2.encoder.layers.{}.attention.out_proj.bias": "encoder_layers.{}.attention.out_proj.bias",
        "wav2vec2.encoder.layers.{}.layer_norm.weight": "encoder_layers.{}.layer_norm.weight",
        "wav2vec2.encoder.layers.{}.layer_norm.bias": "encoder_layers.{}.layer_norm.bias",
        "wav2vec2.encoder.layers.{}.feed_forward.intermediate_dense.weight": "encoder_layers.{}.feed_forward.layers.0.weight",
        "wav2vec2.encoder.layers.{}.feed_forward.intermediate_dense.bias": "encoder_layers.{}.feed_forward.layers.0.bias",
        "wav2vec2.encoder.layers.{}.feed_forward.output_dense.weight": "encoder_layers.{}.feed_forward.layers.3.weight",
        "wav2vec2.encoder.layers.{}.feed_forward.output_dense.bias": "encoder_layers.{}.feed_forward.layers.3.bias",
        "wav2vec2.encoder.layers.{}.final_layer_norm.weight": "encoder_layers.{}.final_layer_norm.weight",
        "wav2vec2.encoder.layers.{}.final_layer_norm.bias": "encoder_layers.{}.final_layer_norm.bias",
        
        # LM head
        "lm_head.weight": "lm_head.weight",
        "lm_head.bias": "lm_head.bias",
    }
    
    # Convert weights
    for pt_name, param in state_dict.items():
        # Convert to numpy
        weight = param.detach().cpu().numpy()
        
        # Find matching MLX name
        mlx_name = None
        for pattern, target in mappings.items():
            if "{}" in pattern:
                # Handle indexed layers
                parts = pattern.split("{}")
                if pt_name.startswith(parts[0]) and pt_name.endswith(parts[-1]):
                    # Extract index
                    middle = pt_name[len(parts[0]):-len(parts[-1])]
                    try:
                        idx = int(middle.split(".")[0])
                        mlx_name = target.format(idx)
                        break
                    except:
                        pass
            elif pt_name == pattern:
                mlx_name = target
                break
        
        if mlx_name:
            # Handle specific layer transformations
            if "conv" in mlx_name and "weight" in mlx_name and len(weight.shape) == 3:
                # Conv1d weights: PyTorch uses (out_channels, in_channels, kernel_size)
                # MLX uses same format
                pass
            elif "attention" in mlx_name and "weight" in mlx_name:
                # Attention weights may need transposing
                if weight.ndim == 2:
                    weight = weight.T
            
            mlx_weights[mlx_name] = mx.array(weight)
            print(f"  {pt_name} -> {mlx_name} {weight.shape}")
        else:
            print(f"  Warning: No mapping for {pt_name}")
    
    return mlx_weights


def convert_wav2vec2_model(
    model_name: str = "facebook/wav2vec2-base-960h",
    output_path: str = "mlx_models/wav2vec2-base-960h-mlx",
    test_conversion: bool = True
) -> bool:
    """Convert Wav2Vec2 model from HuggingFace to MLX.
    
    Args:
        model_name: HuggingFace model ID
        output_path: Output directory for MLX model
        test_conversion: Whether to test the conversion
        
    Returns:
        True if successful
    """
    print(f"Converting {model_name} to MLX format...")
    
    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load PyTorch model
        print("Loading PyTorch model...")
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        
        # Get model config
        config = {
            "hidden_size": model.config.hidden_size,
            "num_hidden_layers": model.config.num_hidden_layers,
            "num_attention_heads": model.config.num_attention_heads,
            "intermediate_size": model.config.intermediate_size,
            "vocab_size": model.config.vocab_size,
            "feat_proj_dropout": model.config.feat_proj_dropout,
            "hidden_dropout": model.config.hidden_dropout_prob,
        }
        
        # Save config
        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # Convert weights
        print("Converting weights...")
        state_dict = model.state_dict()
        mlx_weights = map_wav2vec2_weights(state_dict)
        
        # Save weights
        print(f"Saving to {output_dir / 'weights.npz'}...")
        mx.save(str(output_dir / "weights.npz"), mlx_weights)
        
        # Save processor/tokenizer
        processor.save_pretrained(str(output_dir))
        
        print(f"✓ Model converted successfully!")
        
        # Test conversion if requested
        if test_conversion:
            print("\nTesting conversion...")
            from whisperx.alignment_mlx_v2 import MLXWav2Vec2ForCTC
            
            # Load MLX model
            mlx_model = MLXWav2Vec2ForCTC(config)
            mlx_model.load_weights(mlx_weights)
            
            # Test forward pass
            test_input = mx.random.normal((1, 16000))
            output = mlx_model(test_input)
            print(f"✓ Test forward pass successful! Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        return False


def convert_all_alignment_models(output_dir: str = "~/mlx_models/alignment"):
    """Convert all standard alignment models."""
    output_dir = Path(output_dir).expanduser()
    
    models = [
        ("facebook/wav2vec2-base-960h", "wav2vec2-base-960h-mlx"),
        ("facebook/wav2vec2-large-960h", "wav2vec2-large-960h-mlx"),
        ("facebook/wav2vec2-large-960h-lv60-self", "wav2vec2-large-960h-lv60-mlx"),
        # Add language-specific models
        ("jonatasgrosman/wav2vec2-large-xlsr-53-english", "wav2vec2-xlsr-53-english-mlx"),
        ("jonatasgrosman/wav2vec2-large-xlsr-53-spanish", "wav2vec2-xlsr-53-spanish-mlx"),
        ("jonatasgrosman/wav2vec2-large-xlsr-53-french", "wav2vec2-xlsr-53-french-mlx"),
        ("jonatasgrosman/wav2vec2-large-xlsr-53-german", "wav2vec2-xlsr-53-german-mlx"),
    ]
    
    results = []
    
    for hf_model, mlx_name in models:
        print(f"\n{'='*60}")
        print(f"Converting {hf_model}")
        print('='*60)
        
        success = convert_wav2vec2_model(
            hf_model,
            str(output_dir / mlx_name),
            test_conversion=False  # Skip testing for batch conversion
        )
        
        results.append((hf_model, success))
    
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
        description="Convert Wav2Vec2 models to MLX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single model
  python convert_alignment_models.py --model facebook/wav2vec2-base-960h --output ~/mlx_models/wav2vec2-base-mlx

  # Convert all standard models
  python convert_alignment_models.py --all --output-dir ~/mlx_models/alignment

  # Test existing conversion
  python convert_alignment_models.py --test ~/mlx_models/wav2vec2-base-mlx
""")
    
    parser.add_argument("--model", 
                       help="HuggingFace model ID to convert")
    parser.add_argument("--output",
                       help="Output path for MLX model")
    parser.add_argument("--all", action="store_true",
                       help="Convert all standard models")
    parser.add_argument("--output-dir", default="~/mlx_models/alignment",
                       help="Output directory for --all")
    parser.add_argument("--test",
                       help="Test existing MLX model")
    
    args = parser.parse_args()
    
    if args.test:
        # Test existing model
        print(f"Testing {args.test}...")
        # Implementation would load and test the model
        print("Test functionality not yet implemented")
    elif args.all:
        convert_all_alignment_models(args.output_dir)
    elif args.model and args.output:
        success = convert_wav2vec2_model(args.model, args.output)
        if not success:
            exit(1)
    else:
        parser.error("Specify --model and --output, or use --all")


if __name__ == "__main__":
    main()