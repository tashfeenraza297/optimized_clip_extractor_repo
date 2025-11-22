# =========================================================
# VERSION 2: ALL ENHANCEMENTS
# - Two-tier threshold system
# - Grace period for occlusions
# - Boundary expansion
# - Dense refinement phase
# - Minimum duration filtering
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
        self.num_threads = 4
        
        # VERSION 2 ENHANCED PARAMETERS
        self.initial_skip_fast = 45
        self.initial_skip_accurate = 30
        self.dense_skip_fast = 5
        self.dense_skip_accurate = 3
        
        # Two-tier threshold system
        self.primary_threshold_fast = 0.55
        self.edge_threshold_fast = 0.40
        self.primary_threshold_accurate = 0.65
        self.edge_threshold_accurate = 0.50
        
        # Grace period (in seconds)
        self.max_gap_fast = 1.5
        self.max_gap_accurate = 1.0
        
        # Minimum duration and boundary expansion
        self.min_duration_fast = 2.5
        self.min_duration_accurate = 3.0
        self.boundary_buffer_fast = 1.5
        self.boundary_buffer_accurate = 2.0

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
    
    def sparse_scan(self, mode="fast"):
        """PHASE 1: Sparse scan to identify regions of interest"""
        print(f"\n[V2 PHASE 1] Sparse Scan - {mode.upper()} mode")
        
        cap = cv2.VideoCapture(self.config.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        initial_skip = self.config.initial_skip_fast if mode == "fast" else self.config.initial_skip_accurate
        
        print(f"[INFO] Video: {total_frames} frames @ {fps:.1f} FPS")
        print(f"[INFO] Initial skip: {initial_skip} frames")
        
        all_probs = []
        all_indices = []
        frame_num = 0
        
        pbar = tqdm(total=total_frames, desc="Phase 1: Sparse", unit="frames")
        
        while frame_num < total_frames:
            frames_to_process = []
            frame_positions = []
            
            for _ in range(self.config.batch_size):
                if frame_num >= total_frames:
                    break
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if not ret:
                    break
                
                frames_to_process.append(frame)
                frame_positions.append(frame_num)
                frame_num += initial_skip
            
            if not frames_to_process:
                break
            
            processed_batch = self.preprocess_batch_parallel(frames_to_process)
            predictions = self.model.predict(processed_batch, verbose=0)
            probs = predictions[:, 1]
            
            all_probs.extend(probs)
            all_indices.extend(frame_positions)
            
            pbar.update(frame_positions[-1] - pbar.n)
        
        pbar.close()
        cap.release()
        
        print(f"[✓] Sparse scan: {len(all_probs)} predictions")
        return all_probs, all_indices, fps, total_frames
    
    def identify_roi(self, probs, indices, mode="fast"):
        """Identify regions of interest from sparse scan"""
        primary_threshold = (self.config.primary_threshold_fast if mode == "fast" 
                           else self.config.primary_threshold_accurate)
        
        regions = []
        in_region = False
        region_start = None
        
        for prob, idx in zip(probs, indices):
            if prob >= primary_threshold and not in_region:
                region_start = idx
                in_region = True
            elif prob < primary_threshold and in_region:
                regions.append((region_start, idx))
                in_region = False
        
        if in_region:
            regions.append((region_start, indices[-1]))
        
        print(f"[✓] Identified {len(regions)} region(s) of interest")
        return regions
    
    def dense_refinement(self, regions, fps, total_frames, mode="fast"):
        """PHASE 2: Dense refinement of detected regions"""
        print(f"\n[V2 PHASE 2] Dense Refinement - {mode.upper()} mode")
        
        dense_skip = self.config.dense_skip_fast if mode == "fast" else self.config.dense_skip_accurate
        buffer_frames = int(3 * fps)  # 3 second buffer around regions
        
        cap = cv2.VideoCapture(self.config.video_path)
        all_dense_probs = []
        all_dense_indices = []
        
        for region_idx, (start, end) in enumerate(regions):
            # Expand region with buffer
            expanded_start = max(0, start - buffer_frames)
            expanded_end = min(total_frames - 1, end + buffer_frames)
            
            print(f"  Refining region {region_idx + 1}: frames {expanded_start}-{expanded_end}")
            
            frame_num = expanded_start
            while frame_num <= expanded_end:
                frames_to_process = []
                frame_positions = []
                
                for _ in range(self.config.batch_size):
                    if frame_num > expanded_end:
                        break
                    
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frames_to_process.append(frame)
                    frame_positions.append(frame_num)
                    frame_num += dense_skip
                
                if not frames_to_process:
                    break
                
                processed_batch = self.preprocess_batch_parallel(frames_to_process)
                predictions = self.model.predict(processed_batch, verbose=0)
                probs = predictions[:, 1]
                
                all_dense_probs.extend(probs)
                all_dense_indices.extend(frame_positions)
        
        cap.release()
        print(f"[✓] Dense refinement: {len(all_dense_probs)} predictions")
        
        return all_dense_probs, all_dense_indices
    
    def scan_video_v2(self, mode="fast"):
        """Complete V2 scan with sparse + dense phases"""
        # Phase 1: Sparse scan
        sparse_probs, sparse_indices, fps, total_frames = self.sparse_scan(mode)
        
        # Identify ROIs
        regions = self.identify_roi(sparse_probs, sparse_indices, mode)
        
        if not regions:
            print("[INFO] No regions detected in sparse scan")
            return [], [], fps, total_frames
        
        # Phase 2: Dense refinement
        dense_probs, dense_indices = self.dense_refinement(regions, fps, total_frames, mode)
        
        return dense_probs, dense_indices, fps, total_frames

class ClipExtractor:
    def __init__(self, config, fps):
        self.config = config
        self.fps = fps
        
    def detect_events_v2(self, probs, indices, mode="fast"):
        """VERSION 2: Two-tier threshold with grace period"""
        primary_threshold = (self.config.primary_threshold_fast if mode == "fast" 
                           else self.config.primary_threshold_accurate)
        edge_threshold = (self.config.edge_threshold_fast if mode == "fast" 
                        else self.config.edge_threshold_accurate)
        max_gap_seconds = self.config.max_gap_fast if mode == "fast" else self.config.max_gap_accurate
        min_duration = self.config.min_duration_fast if mode == "fast" else self.config.min_duration_accurate
        
        max_gap_frames = int(max_gap_seconds * self.fps)
        min_frames = int(min_duration * self.fps)
        
        print(f"\n[V2 Detection] {mode.upper()}")
        print(f"  Primary threshold: {primary_threshold}")
        print(f"  Edge threshold: {edge_threshold}")
        print(f"  Max gap: {max_gap_seconds}s ({max_gap_frames} frames)")
        
        events = []
        current_clip_frames = []
        consecutive_low_frames = 0
        in_event = False
        
        for prob, frame_idx in zip(probs, indices):
            # Two-tier logic
            if prob >= primary_threshold:
                # Core detection
                current_clip_frames.append(frame_idx)
                consecutive_low_frames = 0
                in_event = True
                
            elif prob >= edge_threshold and in_event:
                # Edge detection (only when already in event)
                current_clip_frames.append(frame_idx)
                consecutive_low_frames = 0
                
            else:
                # Low confidence
                consecutive_low_frames += 1
                
                # Grace period check
                if consecutive_low_frames >= max_gap_frames and current_clip_frames:
                    # End current clip
                    if len(current_clip_frames) >= min_frames:
                        events.append((current_clip_frames[0], current_clip_frames[-1]))
                    current_clip_frames = []
                    consecutive_low_frames = 0
                    in_event = False
        
        # Handle last clip
        if current_clip_frames and len(current_clip_frames) >= min_frames:
            events.append((current_clip_frames[0], current_clip_frames[-1]))
        
        print(f"[✓] Detected {len(events)} event(s) before boundary expansion")
        return events
    
    def expand_boundaries(self, events, total_frames, mode="fast"):
        """Expand event boundaries for context"""
        buffer_seconds = (self.config.boundary_buffer_fast if mode == "fast" 
                         else self.config.boundary_buffer_accurate)
        buffer_frames = int(buffer_seconds * self.fps)
        
        expanded_events = []
        for start, end in events:
            new_start = max(0, start - buffer_frames)
            new_end = min(total_frames - 1, end + buffer_frames)
            expanded_events.append((new_start, new_end))
        
        print(f"[✓] Expanded boundaries by ±{buffer_seconds}s")
        return expanded_events
    
    def extract_clips(self, events, mode_name, total_frames):
        if not events:
            print(f"[INFO] No {self.config.detection_type} events detected in {mode_name.upper()} mode")
            return
        
        # Expand boundaries
        expanded_events = self.expand_boundaries(events, total_frames, mode_name)
        
        print(f"\n[EXTRACT V2] {mode_name.upper()} Mode: {len(expanded_events)} event(s)")
        
        cap = cv2.VideoCapture(self.config.video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        for idx, (start, end) in enumerate(expanded_events):
            clip_filename = f"V2_{mode_name.upper()}_{self.config.detection_type}_{idx+1}.mp4"
            clip_path = os.path.join(self.config.output_dir, clip_filename)
            
            out = cv2.VideoWriter(clip_path, fourcc, self.fps, (frame_width, frame_height))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            
            frames_written = 0
            for frame_idx in range(start, end + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                frames_written += 1
            
            out.release()
            
            duration = frames_written / self.fps
            start_time = start / self.fps
            end_time = end / self.fps
            
            print(f"  [✓] Clip {idx+1}: {start_time:.1f}s - {end_time:.1f}s (Duration: {duration:.1f}s) → {clip_filename}")
        
        cap.release()
        print(f"[✓] All V2 {mode_name.upper()} clips extracted\n")

def main():
    parser = argparse.ArgumentParser(description='VERSION 2: All Enhancements Clip Extractor')
    parser.add_argument('--model', type=str, required=True, help='Path to detection model')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--mode', type=str, default='both', choices=['fast', 'accurate', 'both'])
    parser.add_argument('--type', type=str, default='accident', choices=['accident', 'fire'])
    parser.add_argument('--batch-size', type=int, default=32)
    
    args = parser.parse_args()
    
    config = Config()
    config.model_path = args.model
    config.video_path = args.video
    config.output_dir = args.output
    config.mode = args.mode
    config.detection_type = args.type
    config.batch_size = args.batch_size
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    print("=\n"*60)
    print(" VERSION 2: ALL ENHANCEMENTS EXTRACTOR")
    print("=\n"*60)
    print(f"Video: {config.video_path}")
    print(f"Detection: {config.detection_type.upper()}")
    print(f"Mode: {config.mode.upper()}")
    print("=\n"*60)
    
    start_time = time.time()
    
    analyzer = OptimizedVideoAnalyzer(config)
    analyzer.load_model()
    
    cap = cv2.VideoCapture(config.video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if config.mode in ['fast', 'both']:
        probs, indices, fps, _ = analyzer.scan_video_v2(mode='fast')
        if probs:
            extractor = ClipExtractor(config, fps)
            events = extractor.detect_events_v2(probs, indices, mode='fast')
            extractor.extract_clips(events, 'fast', total_frames)
    
    if config.mode in ['accurate', 'both']:
        probs, indices, fps, _ = analyzer.scan_video_v2(mode='accurate')
        if probs:
            extractor = ClipExtractor(config, fps)
            events = extractor.detect_events_v2(probs, indices, mode='accurate')
            extractor.extract_clips(events, 'accurate', total_frames)
    
    elapsed = time.time() - start_time
    print("=\n"*60)
    print(f"[V2 COMPLETE] Total time: {elapsed:.2f} seconds")
    print(f"[OUTPUT] Clips saved to: {config.output_dir}")
    print("=\n"*60)

if __name__ == "__main__":
    main()