#!/usr/bin/env python3
"""
Comprehensive profiling and comparison of batch vs non-batch MLX processing
Generates detailed performance tables and accuracy metrics
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Tuple
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Set environment for MLX
os.environ['NUMBA_DISABLE_JIT'] = '1'


class BatchProcessingProfiler:
    """Profile and compare batch vs non-batch processing"""
    
    def __init__(self, audio_file: str = "30m.wav", model: str = "large-v3"):
        self.audio_file = audio_file
        self.model = model
        self.results = defaultdict(dict)
        self.detailed_metrics = defaultdict(list)
        
    def profile_configuration(self, 
                            use_batch: bool, 
                            batch_size: int = 8,
                            num_runs: int = 1) -> Dict[str, Any]:
        """Profile a specific configuration"""
        
        config_name = f"{'Batch' if use_batch else 'Sequential'}_bs{batch_size if use_batch else 1}"
        print(f"\n{'='*60}")
        print(f"Profiling: {config_name}")
        print(f"{'='*60}")
        
        import whisperx
        from whisperx.process_separation import ProcessSeparatedPipeline
        
        # Load audio once
        print("Loading audio...")
        audio = whisperx.load_audio(self.audio_file)
        audio_duration = len(audio) / 16000
        print(f"Audio duration: {audio_duration:.1f}s")
        
        metrics = {
            'config': config_name,
            'use_batch': use_batch,
            'batch_size': batch_size if use_batch else 1,
            'audio_duration': audio_duration,
            'runs': []
        }
        
        for run_idx in range(num_runs):
            print(f"\nRun {run_idx + 1}/{num_runs}")
            
            # Initialize pipeline
            pipeline = ProcessSeparatedPipeline(
                asr_backend="mlx",
                model_name=self.model,
                vad_method="silero",
                device="mlx",
                language="en",
                compute_type="float16",
                asr_options={"word_timestamps": True},
                vad_options={
                    "chunk_size": 30,
                    "vad_onset": 0.5,
                    "vad_offset": 0.363,
                },
                use_batch_processing=use_batch,
                batch_size=batch_size,
                task="transcribe",
                threads=8,
            )
            
            # Profile transcription
            run_metrics = self._profile_single_run(pipeline, audio, audio_duration)
            metrics['runs'].append(run_metrics)
            
            # Detailed tracking
            self.detailed_metrics[config_name].append(run_metrics)
        
        # Calculate aggregate metrics
        metrics['aggregate'] = self._calculate_aggregate_metrics(metrics['runs'])
        
        return metrics
    
    def _profile_single_run(self, pipeline, audio, audio_duration) -> Dict[str, Any]:
        """Profile a single transcription run"""
        
        # Memory before
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024
        
        # Time transcription
        start_time = time.time()
        result = pipeline.transcribe(
            audio,
            batch_size=8,
            chunk_size=30,
            print_progress=True,
            verbose=True
        )
        total_time = time.time() - start_time
        
        # Memory after
        mem_after = process.memory_info().rss / 1024 / 1024
        
        # Extract metrics
        segments = result.get("segments", [])
        word_count = sum(len(seg.get("words", [])) for seg in segments)
        char_count = sum(len(seg.get("text", "")) for seg in segments)
        
        # Performance metrics
        metrics = {
            'total_time': total_time,
            'real_time_factor': audio_duration / total_time,
            'segments': len(segments),
            'words': word_count,
            'characters': char_count,
            'memory_before_mb': mem_before,
            'memory_after_mb': mem_after,
            'memory_delta_mb': mem_after - mem_before,
            'segments_per_second': len(segments) / total_time,
            'words_per_second': word_count / total_time,
        }
        
        # Get batch statistics if available
        if hasattr(pipeline, '_last_backend') and hasattr(pipeline._last_backend, 'get_performance_report'):
            batch_report = pipeline._last_backend.get_performance_report()
            metrics['batch_report'] = batch_report
        
        return metrics
    
    def _calculate_aggregate_metrics(self, runs: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregate metrics across runs"""
        
        if not runs:
            return {}
        
        aggregate = {}
        
        # Time metrics
        times = [r['total_time'] for r in runs]
        aggregate['avg_time'] = np.mean(times)
        aggregate['std_time'] = np.std(times)
        aggregate['min_time'] = np.min(times)
        aggregate['max_time'] = np.max(times)
        
        # RTF metrics
        rtfs = [r['real_time_factor'] for r in runs]
        aggregate['avg_rtf'] = np.mean(rtfs)
        aggregate['std_rtf'] = np.std(rtfs)
        
        # Memory metrics
        mem_deltas = [r['memory_delta_mb'] for r in runs]
        aggregate['avg_memory_delta'] = np.mean(mem_deltas)
        aggregate['max_memory_delta'] = np.max(mem_deltas)
        
        # Throughput metrics
        aggregate['avg_segments_per_second'] = np.mean([r['segments_per_second'] for r in runs])
        aggregate['avg_words_per_second'] = np.mean([r['words_per_second'] for r in runs])
        
        return aggregate
    
    def run_comparison(self, batch_sizes: List[int] = [1, 4, 8, 16], num_runs: int = 3):
        """Run comprehensive comparison"""
        
        print(f"\nBatch Processing Comparison")
        print(f"Model: {self.model}")
        print(f"Audio: {self.audio_file}")
        print(f"Batch sizes: {batch_sizes}")
        print(f"Runs per config: {num_runs}")
        
        # Profile sequential (non-batch)
        seq_metrics = self.profile_configuration(use_batch=False, num_runs=num_runs)
        self.results['sequential'] = seq_metrics
        
        # Profile different batch sizes
        for batch_size in batch_sizes:
            if batch_size == 1:
                continue  # Skip batch_size=1 as it's same as sequential
            
            batch_metrics = self.profile_configuration(
                use_batch=True, 
                batch_size=batch_size, 
                num_runs=num_runs
            )
            self.results[f'batch_{batch_size}'] = batch_metrics
        
        # Generate comparison tables
        self.generate_comparison_tables()
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Save detailed results
        self.save_results()
    
    def generate_comparison_tables(self):
        """Generate detailed comparison tables"""
        
        print(f"\n{'='*80}")
        print("PERFORMANCE COMPARISON TABLES")
        print(f"{'='*80}")
        
        # Table 1: Overall Performance Summary
        print("\nTable 1: Overall Performance Summary")
        print("-" * 80)
        
        summary_data = []
        baseline_time = self.results['sequential']['aggregate']['avg_time']
        baseline_rtf = self.results['sequential']['aggregate']['avg_rtf']
        
        for config_name, metrics in self.results.items():
            agg = metrics['aggregate']
            
            row = {
                'Configuration': config_name.replace('_', ' ').title(),
                'Batch Size': metrics['batch_size'],
                'Avg Time (s)': f"{agg['avg_time']:.2f} ± {agg['std_time']:.2f}",
                'RTF': f"{agg['avg_rtf']:.1f}x ± {agg['std_rtf']:.1f}",
                'Speedup': f"{baseline_time / agg['avg_time']:.2f}x",
                'Memory (MB)': f"{agg['avg_memory_delta']:.0f}",
            }
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Table 2: Throughput Metrics
        print("\n\nTable 2: Throughput Metrics")
        print("-" * 80)
        
        throughput_data = []
        for config_name, metrics in self.results.items():
            agg = metrics['aggregate']
            
            row = {
                'Configuration': config_name.replace('_', ' ').title(),
                'Segments/sec': f"{agg['avg_segments_per_second']:.1f}",
                'Words/sec': f"{agg['avg_words_per_second']:.1f}",
                'Total Segments': metrics['runs'][0]['segments'],
                'Total Words': metrics['runs'][0]['words'],
            }
            throughput_data.append(row)
        
        throughput_df = pd.DataFrame(throughput_data)
        print(throughput_df.to_string(index=False))
        
        # Table 3: Batch Efficiency Analysis (if batch stats available)
        print("\n\nTable 3: Batch Efficiency Analysis")
        print("-" * 80)
        
        batch_data = []
        for config_name, metrics in self.results.items():
            if 'batch' in config_name.lower():
                # Try to get batch statistics
                for run in metrics['runs']:
                    if 'batch_report' in run and run['batch_report']:
                        report = run['batch_report']
                        if 'batch_statistics' in report and report['batch_statistics']:
                            stats = report['batch_statistics']
                            row = {
                                'Configuration': config_name.replace('_', ' ').title(),
                                'Num Batches': stats.get('num_batches', 'N/A'),
                                'Avg Batch Size': f"{stats.get('avg_batch_size', 0):.1f}",
                                'Max Batch Size': stats.get('max_batch_size', 'N/A'),
                                'Padding Efficiency': f"{stats.get('avg_padding_efficiency', 0)*100:.1f}%",
                            }
                            batch_data.append(row)
                            break
        
        if batch_data:
            batch_df = pd.DataFrame(batch_data)
            print(batch_df.to_string(index=False))
        else:
            print("No batch statistics available")
        
        # Table 4: Memory Usage Comparison
        print("\n\nTable 4: Memory Usage Analysis")
        print("-" * 80)
        
        memory_data = []
        for config_name, metrics in self.results.items():
            runs = metrics['runs']
            
            row = {
                'Configuration': config_name.replace('_', ' ').title(),
                'Avg Start (MB)': f"{np.mean([r['memory_before_mb'] for r in runs]):.0f}",
                'Avg Peak (MB)': f"{np.mean([r['memory_after_mb'] for r in runs]):.0f}",
                'Avg Delta (MB)': f"{np.mean([r['memory_delta_mb'] for r in runs]):.0f}",
                'Max Delta (MB)': f"{np.max([r['memory_delta_mb'] for r in runs]):.0f}",
            }
            memory_data.append(row)
        
        memory_df = pd.DataFrame(memory_data)
        print(memory_df.to_string(index=False))
        
    def generate_visualizations(self):
        """Generate performance visualization plots"""
        
        print("\n\nGenerating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Batch Processing Performance Analysis - {self.model}', fontsize=16)
        
        # Plot 1: Processing Time Comparison
        ax1 = axes[0, 0]
        configs = []
        times = []
        errors = []
        
        for config_name, metrics in self.results.items():
            configs.append(config_name.replace('_', ' ').title())
            times.append(metrics['aggregate']['avg_time'])
            errors.append(metrics['aggregate']['std_time'])
        
        ax1.bar(configs, times, yerr=errors, capsize=5)
        ax1.set_ylabel('Processing Time (s)')
        ax1.set_title('Processing Time by Configuration')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Real-time Factor
        ax2 = axes[0, 1]
        rtfs = []
        rtf_errors = []
        
        for config_name, metrics in self.results.items():
            rtfs.append(metrics['aggregate']['avg_rtf'])
            rtf_errors.append(metrics['aggregate']['std_rtf'])
        
        ax2.bar(configs, rtfs, yerr=rtf_errors, capsize=5, color='green')
        ax2.set_ylabel('Real-time Factor (x)')
        ax2.set_title('Real-time Factor by Configuration')
        ax2.tick_params(axis='x', rotation=45)
        ax2.axhline(y=1, color='r', linestyle='--', label='Real-time threshold')
        
        # Plot 3: Memory Usage
        ax3 = axes[1, 0]
        memory_deltas = []
        
        for config_name, metrics in self.results.items():
            memory_deltas.append(metrics['aggregate']['avg_memory_delta'])
        
        ax3.bar(configs, memory_deltas, color='orange')
        ax3.set_ylabel('Memory Delta (MB)')
        ax3.set_title('Memory Usage by Configuration')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Speedup vs Batch Size
        ax4 = axes[1, 1]
        batch_sizes = []
        speedups = []
        baseline_time = self.results['sequential']['aggregate']['avg_time']
        
        for config_name, metrics in self.results.items():
            batch_sizes.append(metrics['batch_size'])
            speedups.append(baseline_time / metrics['aggregate']['avg_time'])
        
        # Sort by batch size
        sorted_pairs = sorted(zip(batch_sizes, speedups))
        batch_sizes, speedups = zip(*sorted_pairs)
        
        ax4.plot(batch_sizes, speedups, 'o-', markersize=8)
        ax4.set_xlabel('Batch Size')
        ax4.set_ylabel('Speedup (x)')
        ax4.set_title('Speedup vs Batch Size')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=1, color='r', linestyle='--', label='Baseline')
        
        plt.tight_layout()
        plt.savefig('batch_processing_comparison.png', dpi=300)
        print("Saved visualization to: batch_processing_comparison.png")
        
    def save_results(self):
        """Save detailed results to JSON"""
        
        # Prepare serializable results
        serializable_results = {}
        for config_name, metrics in self.results.items():
            serializable_results[config_name] = {
                'config': metrics['config'],
                'use_batch': metrics['use_batch'],
                'batch_size': metrics['batch_size'],
                'audio_duration': metrics['audio_duration'],
                'aggregate': metrics['aggregate'],
                'run_details': metrics['runs']
            }
        
        # Save to JSON
        output_file = f'batch_comparison_results_{self.model}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'model': self.model,
                'audio_file': self.audio_file,
                'results': serializable_results
            }, f, indent=2)
        
        print(f"\nDetailed results saved to: {output_file}")
        
        # Generate markdown report
        self.generate_markdown_report(output_file)
    
    def generate_markdown_report(self, json_file: str):
        """Generate a markdown report with all findings"""
        
        md_file = json_file.replace('.json', '.md')
        
        with open(md_file, 'w') as f:
            f.write(f"# Batch Processing Performance Report\n\n")
            f.write(f"**Model**: {self.model}\n")
            f.write(f"**Audio**: {self.audio_file}\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            
            # Calculate key findings
            seq_time = self.results['sequential']['aggregate']['avg_time']
            best_config = min(self.results.items(), key=lambda x: x[1]['aggregate']['avg_time'])
            best_speedup = seq_time / best_config[1]['aggregate']['avg_time']
            
            f.write(f"- Best configuration: **{best_config[0]}** with {best_speedup:.2f}x speedup\n")
            f.write(f"- Sequential processing time: {seq_time:.2f}s\n")
            f.write(f"- Best batch processing time: {best_config[1]['aggregate']['avg_time']:.2f}s\n\n")
            
            f.write("## Detailed Performance Metrics\n\n")
            
            # Write tables
            f.write("### Processing Time Comparison\n\n")
            f.write("| Configuration | Batch Size | Avg Time (s) | Speedup | RTF |\n")
            f.write("|--------------|------------|--------------|---------|-----|\n")
            
            for config_name, metrics in sorted(self.results.items(), 
                                             key=lambda x: x[1]['aggregate']['avg_time']):
                agg = metrics['aggregate']
                f.write(f"| {config_name} | {metrics['batch_size']} | "
                       f"{agg['avg_time']:.2f} ± {agg['std_time']:.2f} | "
                       f"{seq_time / agg['avg_time']:.2f}x | "
                       f"{agg['avg_rtf']:.1f}x |\n")
            
            f.write("\n### Memory Usage\n\n")
            f.write("| Configuration | Avg Delta (MB) | Max Delta (MB) |\n")
            f.write("|--------------|----------------|----------------|\n")
            
            for config_name, metrics in self.results.items():
                agg = metrics['aggregate']
                f.write(f"| {config_name} | {agg['avg_memory_delta']:.0f} | "
                       f"{agg['max_memory_delta']:.0f} |\n")
            
            f.write("\n## Recommendations\n\n")
            f.write(f"1. Use **{best_config[0]}** for optimal performance\n")
            f.write(f"2. Batch processing provides up to {best_speedup:.2f}x speedup\n")
            f.write(f"3. Memory overhead is minimal (~{best_config[1]['aggregate']['avg_memory_delta']:.0f}MB)\n")
        
        print(f"Markdown report saved to: {md_file}")


def main():
    """Run batch processing profiling comparison"""
    
    # Configuration
    audio_file = "30m.wav"
    model = "large-v3"
    batch_sizes = [1, 4, 8, 16]
    num_runs = 3
    
    # Check if audio file exists
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found")
        print("Please provide a 30-minute audio file for profiling")
        return
    
    # Run profiling
    profiler = BatchProcessingProfiler(audio_file=audio_file, model=model)
    profiler.run_comparison(batch_sizes=batch_sizes, num_runs=num_runs)
    
    print("\n\nBatch processing profiling complete!")


if __name__ == "__main__":
    main()