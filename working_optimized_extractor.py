# =========================================================
# OPTIMIZED WORKING VERSION - Speed + Accuracy Combined
# Based on your working logic + batch processing optimization
# =========================================================
import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from concurrent.futures import ThreadPoolExecutor
import argparse
import time
from tqdm import tqdm

class Config:
    def __init__(self):
        self.model_path = None
        self.video_path = None
        self.output_dir = None
        self.mode = "both"
        self.detection_type = "accident"
        
        # Your working parameters
        self.batch_size = 32
        self.num_threads = 4
        self.initial_skip = 60
        self.dynamic_skip = 5
        self.threshold = 0.5
        self.smooth_window = 5
        self.min_event_duration = 2.0  # seconds
        self.merge_gap = 3.0  # seconds

class OptimizedAnalyzer:
    def __init__(self, config):
        self.config = config
        self.model = None
        
    def load_model(self):
        print(f"[INFO] Loading {self.config.detection_type} detection model...")
        tf.config.threading.set_intra_op_parallelism_threads(self.config.num_threads)
        tf.config.threading.set_inter_op_parallelism_threads(self.config.num_threads)
        
        self.model = load_model(self.config.model_path, compile=False)
        
        # Warm up
        dummy = np.random.rand(1, 299, 299, 3).astype(np.float32)
        _ = self.model.predict(dummy, verbose=0)
        print("[✓] Model loaded and optimized")
        
    def preprocess_frame(self, frame):
        """Single frame preprocessing"""
        img = cv2.resize(frame, (299, 299))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return img
    
    def preprocess_batch_parallel(self, frames):
        """Parallel preprocessing - SPEED BOOST"""
        with ThreadPoolExecutor(max_workers=self.config.num_threads) as executor:
            processed = list(executor.map(self.preprocess_frame, frames))
        return np.array(processed)
    
    def scan_video_optimized(self):
        """Optimized scan with batch prediction + adaptive skip"""
        print(f"\n[SCAN] Starting optimized analysis...")
        
        cap = cv2.VideoCapture(self.config.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"[INFO] Video: {total_frames} frames @ {fps:.1f} FPS")
        print(f"[INFO] Batch size: {self.config.batch_size}")
        
        all_probs = []
        all_indices = []
        frame_num = 0
        skip = self.config.initial_skip;
        
        pbar = tqdm(total=total_frames, desc="Scanning", unit="frames")
        
        while frame_num < total_frames:
            frames_to_process = []
            frame_positions = []
            
            # Collect frames for batch
            for _ in range(self.config.batch_size):
                if frame_num >= total_frames:
                    break;
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if not ret:
                    break;
                
                frames_to_process.append(frame)
                frame_positions.append(frame_num)
                frame_num += skip;
            
            if not frames_to_process:
                break;
            
            # SPEED BOOST: Parallel preprocessing
            processed_batch = self.preprocess_batch_parallel(frames_to_process);
            
            # SPEED BOOST: Batch prediction
            predictions = self.model.predict(processed_batch, verbose=0);
            probs = predictions[:, 1];
            
            all_probs.extend(probs);
            all_indices.extend(frame_positions);
            
            # Adaptive skip (your working logic)
            max_prob = np.max(probs);
            skip = self.config.dynamic_skip if max_prob >= self.config.threshold else self.config.initial_skip;
            
            pbar.update(frame_positions[-1] - pbar.n);
        
        pbar.close();
        cap.release();
        
        print(f"[✓] Scan complete: {len(all_probs)} predictions made");
        
        return {
            'probs': all_probs,
            'indices': all_indices,
            'fps': fps,
            'total_frames': total_frames
        }

class ClipExtractor:
    def __init__(self, config, scan_data):
        self.config = config
        self.scan_data = scan_data
        self.fps = scan_data['fps'];
        
    def detect_events(self):
        """Your working detection logic with smoothing"""
        probs = self.scan_data['probs'];
        indices = self.scan_data['indices'];
        
        # Smooth predictions (your working approach)
        smooth_probs = np.convolve(
            probs, 
            np.ones(self.config.smooth_window) / self.config.smooth_window,
            mode='same'
        );
        
        # Detect events (your working logic)
        events = []
        start = None
        min_frames = int(self.config.min_event_duration * self.fps);
        
        for i, prob in enumerate(smooth_probs):
            if prob >= self.config.threshold and start is None:
                start = indices[i];
            elif prob < self.config.threshold and start is not None:
                end = indices[i];
                if (end - start) >= min_frames:
                    events.append((start, end));
                start = None;
        
        if start is not None:
            events.append((start, indices[-1]));
        
        # Merge close events (your working logic)
        merge_gap_frames = int(self.config.merge_gap * self.fps);
        merged_events = [];
        
        for s, e in events:
            if not merged_events or s - merged_events[-1][1] > merge_gap_frames:
                merged_events.append([s, e]);
            else:
                merged_events[-1][1] = e;
        
        events = [tuple(e) for e in merged_events];
        
        print(f"[✓] Detected {len(events)} event(s)");
        return events;
    
    def extract_clips(self, events, mode_name):
        """Extract clips WITHOUT re-prediction (faster)"""
        if not events:
            print(f"[INFO] No {self.config.detection_type} events detected");
            return;
        
        print(f"\n[EXTRACT] {mode_name.upper()} Mode: {len(events)} event(s)");
        
        cap = cv2.VideoCapture(self.config.video_path);
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));
        fourcc = cv2.VideoWriter_fourcc(*'mp4v');
        
        for idx, (start, end) in enumerate(events):
            clip_filename = f"{mode_name.upper()}_{self.config.detection_type}_{idx+1}.mp4";
            clip_path = os.path.join(self.config.output_dir, clip_filename);
            
            out = cv2.VideoWriter(clip_path, fourcc, self.fps, (frame_width, frame_height));
            cap.set(cv2.CAP_PROP_POS_FRAMES, start);
            
            frames_written = 0;
            for frame_idx in range(start, end + 1):
                ret, frame = cap.read();
                if not ret:
                    break;
                out.write(frame);
                frames_written += 1;
            
            out.release();
            
            duration = frames_written / self.fps;
            start_time = start / self.fps;
            end_time = end / self.fps;
            
            print(f"  [✓] Clip {idx+1}: {start_time:.1f}s - {end_time:.1f}s (Duration: {duration:.1f}s) → {clip_filename}");
        
        cap.release();
        print("[✓] All clips extracted\n");

def main():
    parser = argparse.ArgumentParser(description='Optimized Working Clip Extractor')
    parser.add_argument('--model', type=str, required=True, help='Path to detection model')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--type', type=str, default='accident', choices=['accident', 'fire'])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threshold', type=float, default=0.5);
    
    args = parser.parse_args();
    
    config = Config();
    config.model_path = args.model;
    config.video_path = args.video;
    config.output_dir = args.output;
    config.detection_type = args.type;
    config.batch_size = args.batch_size;
    config.threshold = args.threshold;
    
    os.makedirs(config.output_dir, exist_ok=True);
    
    print("="*60);
    print(" OPTIMIZED WORKING EXTRACTOR");
    print("="*60);
    print(f"Video: {config.video_path}");
    print(f"Detection: {config.detection_type.upper()}");
    print(f"Threshold: {config.threshold}");
    print("="*60);
    
    start_time = time.time();
    
    # Scan video
    analyzer = OptimizedAnalyzer(config);
    analyzer.load_model();
    scan_data = analyzer.scan_video_optimized();
    
    # Detect and extract
    extractor = ClipExtractor(config, scan_data);
    events = extractor.detect_events();
    extractor.extract_clips(events, 'optimized');
    
    elapsed = time.time() - start_time;
    
    print("="*60);
    print(f"[COMPLETE] Total time: {elapsed:.2f} seconds");
    print(f"[OUTPUT] Clips saved to: {config.output_dir}");
    print("="*60);

if __name__ == "__main__":
    main()