# =========================================================
# VERSION 1: SIMPLE OPTIMIZED (My Original Optimization Plan)
# - Stricter gap detection (1s instead of 3s)
# - Higher threshold for accurate mode
# - Frame-level filtering
# =========================================================
import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from concurrent.futures import ThreadPoolExecutor
import pickle
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
        self.batch_size = 32
        self.fast_skip = 60
        self.dense_skip = 3
        self.num_threads = 4
        
        # VERSION 1 SPECIFIC PARAMETERS
        self.threshold_fast = 0.5
        self.threshold_accurate = 0.7
        self.max_gap_seconds = 1.0  # Stricter: 1 second gap
        self.min_event_duration = 2.5

class OptimizedVideoAnalyzer:
    def __init__(self, config):
        self.config = config
        self.model = None
        
    def load_model(self):
        print(f"[INFO] Loading {self.config.detection_type} detection model...")
        tf.config.threading.set_intra_op_parallelism_threads(self.config.num_threads)
        tf.config.threading.set_inter_op_parallelism_threads(self.config.num_threads)
        
        self.model = load_model(self.config.model_path, compile=False)
        dummy = np.random.rand(1, 299, 299, 3).astype(np.float32)
        _ = self.model.predict(dummy, verbose=0)
        print("[✓] Model loaded and optimized")
        
    def preprocess_frame(self, frame):
        img = cv2.resize(frame, (299, 299))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return img
    
    def preprocess_batch_parallel(self, frames):
        with ThreadPoolExecutor(max_workers=self.config.num_threads) as executor:
            processed = list(executor.map(self.preprocess_frame, frames))
        return np.array(processed)
    
    def scan_video_optimized(self):
        print(f"\n[VERSION 1] Simple Optimized Scan")
        print(f"[MODE] Detection Type: {self.config.detection_type.upper()}")
        
        cap = cv2.VideoCapture(self.config.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"[INFO] Video: {total_frames} frames @ {fps:.1f} FPS")
        print(f"[INFO] Batch size: {self.config.batch_size}")
        
        all_probs = []
        all_indices = []
        frame_num = 0
        skip = self.config.fast_skip;
        
        pbar = tqdm(total=total_frames, desc="V1 Scanning", unit="frames")
        
        while frame_num < total_frames:
            frames_to_process = []
            frame_positions = []
            
            for _ in range(self.config.batch_size):
                if frame_num >= total_frames:
                    break;
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if not ret:
                    break;
                
                frames_to_process.append(frame);
                frame_positions.append(frame_num);
                frame_num += skip;
            
            if not frames_to_process:
                break;
            
            processed_batch = self.preprocess_batch_parallel(frames_to_process);
            predictions = self.model.predict(processed_batch, verbose=0);
            probs = predictions[:, 1];
            
            all_probs.extend(probs);
            all_indices.extend(frame_positions);
            
            max_prob = np.max(probs);
            skip = self.config.dense_skip if max_prob >= 0.5 else self.config.fast_skip;
            
            pbar.update(frame_positions[-1] - pbar.n);
        
        pbar.close();
        cap.release();
        
        print(f"[✓] Scan complete: {len(all_probs)} predictions made")
        
        cache_data = {
            'probs': all_probs,
            'indices': all_indices,
            'fps': fps,
            'total_frames': total_frames,
            'detection_type': self.config.detection_type
        };
        
        cache_path = os.path.join(self.config.output_dir, f'cache_v1_{self.config.detection_type}.pkl');
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f);
        
        return cache_data;

class ClipExtractor:
    def __init__(self, config, cache_data):
        self.config = config;
        self.cache_data = cache_data;
        self.fps = cache_data['fps'];
        
    def detect_events_v1(self, probs, indices, mode="fast"):
        """VERSION 1: Simple threshold with strict gap detection"""
        threshold = self.config.threshold_fast if mode == "fast" else self.config.threshold_accurate;
        max_gap_frames = int(self.config.max_gap_seconds * self.fps);
        min_frames = int(self.config.min_event_duration * self.fps);
        
        print(f"\n[V1 Detection] Mode: {mode.upper()}, Threshold: {threshold}, Max Gap: {self.config.max_gap_seconds}s")
        
        events = [];
        current_clip_frames = [];
        gap_counter = 0;
        
        for i, (prob, frame_idx) in enumerate(zip(probs, indices)):
            if prob >= threshold:
                # High confidence frame
                current_clip_frames.append(frame_idx);
                gap_counter = 0;
            else:
                # Low confidence frame
                gap_counter += 1;
                
                if gap_counter >= max_gap_frames and current_clip_frames:
                    # Gap too large - end current clip
                    if len(current_clip_frames) >= min_frames:
                        events.append((current_clip_frames[0], current_clip_frames[-1]));
                    current_clip_frames = [];
                    gap_counter = 0;
        
        # Handle last clip
        if current_clip_frames and len(current_clip_frames) >= min_frames:
            events.append((current_clip_frames[0], current_clip_frames[-1]));
        
        print(f"[V1] Detected {len(events)} event(s)");
        return events;
    
    def extract_clips(self, events, mode_name):
        if not events:
            print(f"[INFO] No {self.config.detection_type} events detected in {mode_name.upper()} mode");
            return;
        
        print(f"\n[EXTRACT V1] {mode_name.upper()} Mode: {len(events)} event(s)");
        
        cap = cv2.VideoCapture(self.config.video_path);
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));
        fourcc = cv2.VideoWriter_fourcc(*'mp4v');
        
        for idx, (start, end) in enumerate(events):
            clip_filename = f"V1_{mode_name.upper()}_{self.config.detection_type}_{idx+1}.mp4";
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
        print(f"[✓] All V1 {mode_name.upper()} clips extracted\n");

def main():
    parser = argparse.ArgumentParser(description='VERSION 1: Simple Optimized Clip Extractor')
    parser.add_argument('--model', type=str, required=True, help='Path to detection model')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--mode', type=str, default='both', choices=['fast', 'accurate', 'both'])
    parser.add_argument('--type', type=str, default='accident', choices=['accident', 'fire'])
    parser.add_argument('--batch-size', type=int, default=32)
    
    args = parser.parse_args();
    
    config = Config();
    config.model_path = args.model;
    config.video_path = args.video;
    config.output_dir = args.output;
    config.mode = args.mode;
    config.detection_type = args.type;
    config.batch_size = args.batch_size;
    
    os.makedirs(config.output_dir, exist_ok=True);
    
    print("="*60);
    print(" VERSION 1: SIMPLE OPTIMIZED EXTRACTOR");
    print("="*60);
    print(f"Video: {config.video_path}");
    print(f"Detection: {config.detection_type.upper()}");
    print(f"Mode: {config.mode.upper()}");
    print("="*60);
    
    start_time = time.time();
    
    analyzer = OptimizedVideoAnalyzer(config);
    analyzer.load_model();
    cache_data = analyzer.scan_video_optimized();
    
    extractor = ClipExtractor(config, cache_data);
    probs = cache_data['probs'];
    indices = cache_data['indices'];
    
    if config.mode in ['fast', 'both']:
        events = extractor.detect_events_v1(probs, indices, mode='fast');
        extractor.extract_clips(events, 'fast');
    
    if config.mode in ['accurate', 'both']:
        events = extractor.detect_events_v1(probs, indices, mode='accurate');
        extractor.extract_clips(events, 'accurate');
    
    elapsed = time.time() - start_time;
    print("="*60);
    print(f"[V1 COMPLETE] Total time: {elapsed:.2f} seconds");
    print(f"[OUTPUT] Clips saved to: {config.output_dir}");
    print("="*60);

if __name__ == "__main__":
    main()