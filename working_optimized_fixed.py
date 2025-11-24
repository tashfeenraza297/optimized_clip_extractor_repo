# Working Optimized Fixed

# Constants
INITIAL_SKIP_FRAMES = 15  # Reduced from 60
MERGE_GAP = 6.0  # Increased from 3.0
MIN_EVENT_DURATION = 1.5  # Reduced from 2.0

# Conditional smoothing parameters
window_size = 3  # Applies if enough predictions

# Auto-tuned threshold logic
def auto_tune_threshold(max_prob):
    if max_prob < 0.6:
        return 0.35
    elif max_prob < 0.8:
        return 0.45
    else:
        return 0.5  # or user-specified

# Debugging mode
def debug_statistics(predictions):
    num_predictions = len(predictions)
    min_prob = min(predictions) if predictions else 0
    max_prob = max(predictions) if predictions else 0
    avg_prob = sum(predictions) / len(predictions) if predictions else 0
    tuned_threshold = auto_tune_threshold(max_prob)
    print(f"Number of predictions: {num_predictions}")
    print(f"Min probability: {min_prob}")
    print(f"Max probability: {max_prob}")
    print(f"Average probability: {avg_prob}")
    print(f"Auto-tuned threshold: {tuned_threshold}")

# Include all speed optimizations from working_optimized_extractor.py
# Batch prediction, multithreading, adaptive skip, etc.
# ... (remaining code logic with optimizations)

# After scanning is complete, call the debug_statistics function
# debug_statistics(prediction_values)
