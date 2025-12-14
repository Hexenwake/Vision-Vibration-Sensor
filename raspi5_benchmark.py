"""
Script 3: Raspberry Pi 5 Benchmark Script
Evaluates all TFLite models on Raspberry Pi 5 with the prepared test dataset
"""

import os
import numpy as np
import json
import time
from pathlib import Path
import platform

# Check if TFLite is available
try:
    import tensorflow.lite as tflite
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        print("Using tflite_runtime")
    except ImportError:
        print("ERROR: Neither tensorflow nor tflite_runtime found!")
        print("Install with: pip install tflite-runtime")
        exit(1)

# ==========================
# CONFIGURATION
# ==========================
DEPLOYMENT_DIR = "./raspi5_deployment"
RESULTS_DIR = os.path.join(DEPLOYMENT_DIR, "benchmark_results")
N_RUNS = 5
MODEL_NAMES = ['Baseline', 'Opt-V1', 'Opt-V2', 'Opt-V3', 'Opt-V4']
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 50  # More iterations for accurate timing on RPi5

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================
# SYSTEM INFO
# ==========================
def get_system_info():
    """Get Raspberry Pi system information"""
    info = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'machine': platform.machine()
    }
    
    # Try to get CPU info
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            for line in cpuinfo.split('\n'):
                if 'Model' in line:
                    info['model'] = line.split(':')[1].strip()
                    break
    except:
        info['model'] = 'Unknown'
    
    # Try to get CPU frequency
    try:
        with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq', 'r') as f:
            freq = int(f.read().strip()) / 1000  # Convert to MHz
            info['cpu_freq_mhz'] = freq
    except:
        info['cpu_freq_mhz'] = 'Unknown'
    
    # Try to get temperature
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp = int(f.read().strip()) / 1000  # Convert to Celsius
            info['temperature_c'] = temp
    except:
        info['temperature_c'] = 'Unknown'
    
    return info


# ==========================
# TFLITE INFERENCE
# ==========================
class TFLiteModel:
    """Wrapper for TFLite model inference"""
    
    def __init__(self, model_path, num_threads=4):
        # Use num_threads for RPi5 optimization
        self.interpreter = tflite.Interpreter(
            model_path=model_path,
            num_threads=num_threads
        )
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get quantization parameters
        self.input_dtype = self.input_details[0]['dtype']
        self.output_dtype = self.output_details[0]['dtype']
        
        self.input_scale, self.input_zero_point = self.input_details[0]['quantization']
        self.output_scale, self.output_zero_point = self.output_details[0]['quantization']
    
    def predict_single(self, x):
        """Predict single sample"""
        input_data = x.reshape(1, *x.shape)
        
        # Quantize input if INT8
        if self.input_dtype == np.int8:
            input_data = input_data / self.input_scale + self.input_zero_point
            input_data = np.clip(input_data, -128, 127).astype(np.int8)
        else:
            input_data = input_data.astype(self.input_dtype)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Dequantize output if INT8
        if self.output_dtype == np.int8:
            output_data = (output_data.astype(np.float32) - self.output_zero_point) * self.output_scale
        
        return output_data[0, 0]
    
    def predict_batch(self, X):
        """Predict batch of samples"""
        predictions = []
        for i in range(len(X)):
            pred = self.predict_single(X[i])
            predictions.append(pred)
        return np.array(predictions)


# ==========================
# BENCHMARK FUNCTIONS
# ==========================
def measure_inference_time_single(model, X_test, n_iterations=BENCHMARK_ITERATIONS):
    """Measure single-sample inference time (most realistic for edge devices)"""
    
    # Warmup
    print("      Warming up...", end='', flush=True)
    for _ in range(WARMUP_ITERATIONS):
        _ = model.predict_single(X_test[0])
    print(" Done")
    
    # Benchmark
    print(f"      Benchmarking {n_iterations} iterations...", end='', flush=True)
    times = []
    
    for i in range(n_iterations):
        start = time.perf_counter()
        _ = model.predict_single(X_test[i % len(X_test)])
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    print(" Done")
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'median_ms': np.median(times)
    }


def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    # R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # MAPE
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'mape': float(mape)
    }


def get_file_size_mb(file_path):
    """Get file size in MB"""
    return os.path.getsize(file_path) / (1024 * 1024)


# ==========================
# MAIN BENCHMARK
# ==========================
def benchmark_all_models(num_threads=4):
    """Benchmark all TFLite models"""
    
    print("=" * 80)
    print("RASPBERRY PI 5 - TFLITE MODEL BENCHMARK")
    print("=" * 80)
    
    # Get system info
    print("\n[1/4] Gathering system information...")
    system_info = get_system_info()
    print(f"  Model: {system_info.get('model', 'Unknown')}")
    print(f"  CPU Freq: {system_info.get('cpu_freq_mhz', 'Unknown')} MHz")
    print(f"  Temperature: {system_info.get('temperature_c', 'Unknown')}°C")
    print(f"  Threads: {num_threads}")
    
    # Load test data
    print("\n[2/4] Loading test data...")
    X_test = np.load(os.path.join(DEPLOYMENT_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(DEPLOYMENT_DIR, 'y_test.npy'))
    
    with open(os.path.join(DEPLOYMENT_DIR, 'test_metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    print(f"  Test samples: {len(X_test)}")
    print(f"  Input shape: {X_test.shape}")
    
    # Benchmark all models
    print("\n[3/4] Benchmarking models...")
    all_results = []
    
    tflite_dir = os.path.join(DEPLOYMENT_DIR, 'tflite_models')
    
    for model_name in MODEL_NAMES:
        print(f"\n{'='*80}")
        print(f"Model: {model_name}")
        print(f"{'='*80}")
        
        model_dir = os.path.join(tflite_dir, model_name)
        
        if not os.path.exists(model_dir):
            print(f"  ⚠ Warning: Model directory not found: {model_dir}")
            continue
        
        for run in range(N_RUNS):
            run_dir = os.path.join(model_dir, f'run_{run}')
            
            if not os.path.exists(run_dir):
                print(f"  ⚠ Warning: Run directory not found: {run_dir}")
                continue
            
            print(f"\n  Run {run + 1}/{N_RUNS}")
            
            # Test each quantization type
            for quant_type in ['fp32', 'fp16', 'int8']:
                model_path = os.path.join(run_dir, f'model_{quant_type}.tflite')
                
                if not os.path.exists(model_path):
                    print(f"    ⚠ {quant_type.upper()}: Model not found")
                    continue
                
                print(f"    Testing {quant_type.upper()}:")
                
                try:
                    # Load model
                    print(f"      Loading model...")
                    tflite_model = TFLiteModel(model_path, num_threads=num_threads)
                    
                    # Measure inference time
                    print(f"      Measuring inference time...")
                    timing = measure_inference_time_single(tflite_model, X_test)
                    
                    # Get predictions
                    print(f"      Getting predictions on full test set...")
                    y_pred = tflite_model.predict_batch(X_test)
                    
                    # Calculate metrics
                    metrics = calculate_metrics(y_test, y_pred)
                    
                    # Get model size
                    model_size = get_file_size_mb(model_path)
                    
                    # Get temperature after inference
                    try:
                        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                            temp_after = int(f.read().strip()) / 1000
                    except:
                        temp_after = None
                    
                    result = {
                        'model_name': model_name,
                        'run': run,
                        'quantization': quant_type,
                        'metrics': metrics,
                        'timing': timing,
                        'model_size_mb': model_size,
                        'temperature_after_c': temp_after
                    }
                    
                    all_results.append(result)
                    
                    # Print results
                    print(f"      ✓ Results:")
                    print(f"        MAE: {metrics['mae']:.4f} Hz")
                    print(f"        RMSE: {metrics['rmse']:.4f} Hz")
                    print(f"        R²: {metrics['r2']:.4f}")
                    print(f"        MAPE: {metrics['mape']:.2f}%")
                    print(f"        Inference Time: {timing['mean_ms']:.3f} ± {timing['std_ms']:.3f} ms")
                    print(f"        Min/Max: {timing['min_ms']:.3f} / {timing['max_ms']:.3f} ms")
                    print(f"        Model Size: {model_size:.2f} MB")
                    if temp_after:
                        print(f"        Temperature: {temp_after:.1f}°C")
                    
                except Exception as e:
                    print(f"      ✗ Error: {e}")
    
    # Save results
    print("\n[4/4] Saving results...")
    
    # Save detailed results
    results_file = os.path.join(RESULTS_DIR, 'benchmark_results.json')
    with open(results_file, 'w') as f:
        output = {
            'system_info': system_info,
            'test_metadata': metadata,
            'benchmark_config': {
                'warmup_iterations': WARMUP_ITERATIONS,
                'benchmark_iterations': BENCHMARK_ITERATIONS,
                'num_threads': num_threads
            },
            'results': all_results
        }
        json.dump(output, f, indent=2)
    
    print(f"  ✓ Detailed results: {results_file}")
    
    # Create summary CSV
    import csv
    
    csv_file = os.path.join(RESULTS_DIR, 'benchmark_summary.csv')
    with open(csv_file, 'w', newline='') as f:
        fieldnames = ['Model', 'Run', 'Quantization', 'MAE (Hz)', 'RMSE (Hz)', 
                     'R²', 'MAPE (%)', 'Inference Mean (ms)', 'Inference Std (ms)',
                     'Inference Min (ms)', 'Inference Max (ms)', 'Model Size (MB)']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                'Model': r['model_name'],
                'Run': r['run'],
                'Quantization': r['quantization'].upper(),
                'MAE (Hz)': f"{r['metrics']['mae']:.4f}",
                'RMSE (Hz)': f"{r['metrics']['rmse']:.4f}",
                'R²': f"{r['metrics']['r2']:.4f}",
                'MAPE (%)': f"{r['metrics']['mape']:.2f}",
                'Inference Mean (ms)': f"{r['timing']['mean_ms']:.3f}",
                'Inference Std (ms)': f"{r['timing']['std_ms']:.3f}",
                'Inference Min (ms)': f"{r['timing']['min_ms']:.3f}",
                'Inference Max (ms)': f"{r['timing']['max_ms']:.3f}",
                'Model Size (MB)': f"{r['model_size_mb']:.2f}"
            })
    
    print(f"  ✓ Summary CSV: {csv_file}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE - SUMMARY")
    print("=" * 80)
    
    # Group by quantization type
    for quant in ['fp32', 'fp16', 'int8']:
        quant_results = [r for r in all_results if r['quantization'] == quant]
        if quant_results:
            times = [r['timing']['mean_ms'] for r in quant_results]
            maes = [r['metrics']['mae'] for r in quant_results]
            
            print(f"\n{quant.upper()}:")
            print(f"  Inference Time: {np.mean(times):.3f} ± {np.std(times):.3f} ms")
            print(f"  MAE: {np.mean(maes):.4f} ± {np.std(maes):.4f} Hz")
            print(f"  Models tested: {len(quant_results)}")
    
    print("\n" + "=" * 80)
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 80)


# ==========================
# MAIN EXECUTION
# ==========================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark TFLite models on Raspberry Pi 5')
    parser.add_argument('--threads', type=int, default=4, 
                       help='Number of CPU threads to use (default: 4)')
    parser.add_argument('--iterations', type=int, default=50,
                       help='Number of benchmark iterations (default: 50)')
    
    args = parser.parse_args()
    
    BENCHMARK_ITERATIONS = args.iterations
    
    print(f"\nConfiguration:")
    print(f"  Threads: {args.threads}")
    print(f"  Iterations: {args.iterations}")
    print()
    
    benchmark_all_models(num_threads=args.threads)
