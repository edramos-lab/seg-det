import os
import time
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.data.dataset import StrawberrySegmentationDataset, get_transforms
import onnxruntime as ort

# Try to import TensorRT
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("TensorRT not available. Skipping TensorRT benchmarking.")

def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_test_dataset(config):
    test_dataset = StrawberrySegmentationDataset(
        root_path=config['dataset']['root_path'],
        split=config['dataset']['test_split'],
        transform=get_transforms(config, "val"),
        num_classes=config['dataset']['num_classes']
    )
    return test_dataset

def run_onnx_inference(onnx_path, dataset, device="cpu"):
    print(f"\nRunning ONNX inference on {len(dataset)} test images...")
    session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    times = []
    detections = []
    fps_list = []
    preprocess_times = []
    postprocess_times = []
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        img = sample['image'].unsqueeze(0).cpu().numpy()  # (1, C, H, W)
        # Preprocessing time (simulate if needed)
        t0 = time.time()
        # (already preprocessed by dataset)
        t1 = time.time()
        # Inference
        t2 = time.time()
        output = session.run(None, {input_name: img})[0]
        t3 = time.time()
        # Postprocessing (argmax for segmentation)
        pred = np.argmax(output, axis=1)
        t4 = time.time()
        times.append(t3-t2)
        preprocess_times.append(t1-t0)
        postprocess_times.append(t4-t3)
        # Count detections (non-background pixels)
        detections.append(np.count_nonzero(pred))
        if t3-t2 > 0:
            fps_list.append(1.0/(t3-t2))
    return {
        'inference_times': np.array(times),
        'fps': np.array(fps_list),
        'detections': np.array(detections),
        'preprocess_times': np.array(preprocess_times),
        'postprocess_times': np.array(postprocess_times)
    }

def run_tensorrt_inference(trt_path, dataset):
    if not TENSORRT_AVAILABLE:
        print("TensorRT not available.")
        return None
    print(f"\nRunning TensorRT inference on {len(dataset)} test images...")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(trt_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    input_shape = engine.get_binding_shape(0)
    # Fix for NumPy deprecation warning
    try:
        dtype = trt.nptype(engine.get_binding_dtype(0))
    except AttributeError:
        # Fallback for newer NumPy versions
        dtype = np.float32
    times = []
    detections = []
    fps_list = []
    preprocess_times = []
    postprocess_times = []
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        img = sample['image'].unsqueeze(0).cpu().numpy().astype(dtype)
        # Ensure array is contiguous
        img = np.ascontiguousarray(img)
        # Preprocessing time (simulate if needed)
        t0 = time.time()
        # (already preprocessed by dataset)
        t1 = time.time()
        # Allocate device memory
        d_input = cuda.mem_alloc(img.nbytes)
        output_shape = (1, engine.get_binding_shape(1)[1], engine.get_binding_shape(1)[2], engine.get_binding_shape(1)[3])
        output = np.empty(output_shape, dtype=np.float32)
        d_output = cuda.mem_alloc(output.nbytes)
        bindings = [int(d_input), int(d_output)]
        # Inference
        t2 = time.time()
        cuda.memcpy_htod(d_input, img)
        context.execute_v2(bindings)
        cuda.memcpy_dtoh(output, d_output)
        t3 = time.time()
        # Postprocessing (argmax for segmentation)
        pred = np.argmax(output, axis=1)
        t4 = time.time()
        times.append(t3-t2)
        preprocess_times.append(t1-t0)
        postprocess_times.append(t4-t3)
        detections.append(np.count_nonzero(pred))
        if t3-t2 > 0:
            fps_list.append(1.0/(t3-t2))
        d_input.free()
        d_output.free()
    return {
        'inference_times': np.array(times),
        'fps': np.array(fps_list),
        'detections': np.array(detections),
        'preprocess_times': np.array(preprocess_times),
        'postprocess_times': np.array(postprocess_times)
    }

def plot_results(results, label, outdir):
    os.makedirs(outdir, exist_ok=True)
    # Inference time distribution
    plt.figure()
    plt.hist(results['inference_times']*1000, bins=30, alpha=0.7)
    plt.xlabel('Inference Time (ms)')
    plt.ylabel('Count')
    plt.title(f'{label} Inference Time Distribution')
    plt.savefig(os.path.join(outdir, f'{label}_inference_time_hist.png'))
    # FPS over time
    plt.figure()
    plt.plot(results['fps'])
    plt.xlabel('Image Index')
    plt.ylabel('FPS')
    plt.title(f'{label} FPS Over Time')
    plt.savefig(os.path.join(outdir, f'{label}_fps_over_time.png'))
    # Detections per image
    plt.figure()
    plt.hist(results['detections'], bins=30, alpha=0.7)
    plt.xlabel('Detections per Image')
    plt.ylabel('Count')
    plt.title(f'{label} Detections per Image')
    plt.savefig(os.path.join(outdir, f'{label}_detections_per_image.png'))
    # Time breakdown
    plt.figure()
    avg_pre = np.mean(results['preprocess_times'])*1000
    avg_inf = np.mean(results['inference_times'])*1000
    avg_post = np.mean(results['postprocess_times'])*1000
    plt.bar(['Preprocessing', 'Inference', 'Postprocessing'], [avg_pre, avg_inf, avg_post])
    plt.ylabel('Time (ms)')
    plt.title(f'{label} Time Breakdown')
    plt.savefig(os.path.join(outdir, f'{label}_time_breakdown.png'))
    plt.close('all')

def print_performance(results, label):
    print(f"\n==== {label} Performance Metrics ====")
    print(f"Average Inference Time: {np.mean(results['inference_times'])*1000:.2f} ms")
    print(f"Average FPS: {np.mean(results['fps']):.2f}")
    print(f"Average Detections per Image: {np.mean(results['detections']):.2f}")
    print(f"Total Images: {len(results['inference_times'])}")

def main():
    config = load_config()
    test_dataset = get_test_dataset(config)
    onnx_path = os.path.join('models', 'model.onnx')
    trt_path = os.path.join('models', 'model.trt')
    outdir = 'results/benchmark'
    # ONNX
    onnx_results = run_onnx_inference(onnx_path, test_dataset)
    plot_results(onnx_results, 'ONNX', outdir)
    print_performance(onnx_results, 'ONNX')
    # TensorRT
    if TENSORRT_AVAILABLE and os.path.exists(trt_path):
        trt_results = run_tensorrt_inference(trt_path, test_dataset)
        if trt_results:
            plot_results(trt_results, 'TensorRT', outdir)
            print_performance(trt_results, 'TensorRT')
    else:
        print("TensorRT engine not found or not available. Skipping TensorRT benchmarking.")

if __name__ == "__main__":
    main() 