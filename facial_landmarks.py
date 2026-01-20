import os
import cv2
import json
import logging
import numpy as np
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.signal import butter, detrend, hilbert, filtfilt

logger = logging.getLogger(__name__)

# paths
model_path = os.path.join("data", "face_landmarker.task")
output_path = "data/annotated.mp4"

# MediaPipe setup
BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

# regions to track
REGIONS = ["forehead", "left_cheek", "right_cheek", "mid_region"]

# processing parameters
WINDOW_SIZE = 30   # frames per window 
STEP_SIZE = 10
SMOOTH_WINDOW = 3  # smoothing heart rate

def smooth_hr(hr_list, window=3):
    """
    smoothen heart rate
    """
    if len(hr_list) < window:
        return hr_list
    return np.convolve(hr_list, np.ones(window)/window, mode='valid')

def sliding_window_segments(signal, window_size, step_size):
    if len(signal) <= window_size:
        return [signal]  # short signals, single segment
    segments = []
    for start in range(0, len(signal) - window_size + 1, step_size):
        segments.append(signal[start:start+window_size])
    return segments

def normalize_signal(signal):
    std = np.std(signal)
    if std == 0:
        return signal
    return (signal - np.mean(signal)) / std

def bandpass_filter(signal, fps, low=0.7, high=4.0):
    """
    use butterworth bandpass to filter through freqs
    """
    nyquist = 0.5 * fps
    low_cut = low / nyquist
    high_cut = high / nyquist
    b, a = butter(N=3, Wn=[low_cut, high_cut], btype='band')
    return filtfilt(b, a, signal)

def calc_heart_rate(signal, fps, hr_min=40, hr_max=180):
    """
    extract heart rate using fft
    """
    fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1/fps)
    hr_range = (freqs*60 >= hr_min) & (freqs*60 <= hr_max)
    if not np.any(hr_range):
        return None
    fft_valid = np.abs(fft)[hr_range]
    freqs_valid = freqs[hr_range]
    hr_bpm = freqs_valid[np.argmax(fft_valid)] * 60
    return hr_bpm

def extract_mean_values(frame, region_pts):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    pts = np.array([region_pts], dtype=np.int32)
    cv2.fillPoly(mask, pts, 255)
    region_pixels = frame[mask == 255]
    if region_pixels.size == 0:
        return None
    mean_bgr = region_pixels.mean(axis=0)
    return mean_bgr[::-1]  # convert BGR → RGB

def get_regions(landmarks, h, w):
    """
    get regions from detected face and divide into relevant regions
    """
    regions = {name: [] for name in REGIONS}
    indices = {
        "left_cheek": [234, 93, 132, 58, 172],
        "right_cheek": [454, 323, 361, 288, 397],
        "mid_region": [1, 168, 197, 5, 4],
        "forehead": [10, 338, 297, 332, 284]
    }
    for name, idx_list in indices.items():
        points = [(int(np.clip(landmarks.landmark[i].x*w,0,w-1)),
                   int(np.clip(landmarks.landmark[i].y*h,0,h-1))) for i in idx_list]
        if len(points) >= 3:
            regions[name] = points
    return regions

def export_hr(raw_hr, smooth_hr, filepath):
    df = pd.DataFrame({
        'raw_hr': raw_hr[:len(smooth_hr)],
        'smooth_hr': smooth_hr
    })

    df.to_csv(filepath, index=False)
    print(f"Heart rates exported successfully to {filepath}")

def compute_signal_quality(segment, fps, rate_band=(0.7, 4.0)):
    """
    compute variance, snr, special entropy, band energy for
    each window
    """
    metrics = {}

    # variance
    metrics['variance'] = np.var(segment)

    # fft
    fft = np.fft.rfft(segment)
    fft_mag = np.abs(fft)
    freqs = np.fft.rfftfreq(len(segment), 1/fps)

    # band energy
    band_idx = (freqs >= rate_band[0]) & (freqs <= rate_band[1])
    if fft_mag.sum() > 0:
        metrics['band_energy'] = fft_mag[band_idx].sum() / fft_mag.sum()
    else:
        metrics['band_energy'] = 0

    # snr
    peak = fft_mag.max()
    noise = (fft_mag.sum() - peak) / (len(fft_mag)-1)
    metrics['snr'] = peak / (noise + 1e-8)

    # spectral entropy
    psd = fft_mag**2
    psd_norm = psd / (psd.sum() + 1e-8)
    metrics['spectral_entropy'] = -(psd_norm * np.log(psd_norm + 1e-8)).sum()

    return metrics

def calc_correlation(lc, rc, window, step):
    """
    calculate the region correlation between different face regions
    using pearson's correlation and sliding window technique
    lc = left cheek, rc = right cheek, window = window size, step = step per window
    """
    correlations = []
    for i in range(0, len(lc) - window + 1, step):
        left_cheek = lc[i:i+window]
        right_cheek = rc[i:i+window]

        if np.std(left_cheek) == 0 or np.std(right_cheek) == 0:
            continue

        correlations.append(np.corrcoef(left_cheek, right_cheek)[0, 1])

    if len(correlations) == 0:
        return {
            "mean_correlation": np.nan,
            "correlation_variance": np.nan
        } 
    
    return {
            "mean_correlation": np.mean(correlations),
            "correlation_variance": np.var(correlations)
        }

def extract_phase(signal):
    """
    extract phases from signal using hilbert's transform
    """
    analytic = hilbert(signal)
    return np.unwrap(np.angle(analytic))

def phase_coherence(sig1, sig2, window, step):
    """
    calculate phase coherence between two face regions using hilbert's transform
    sig1 = signalA, sig2 = sig2, window = window size, step = step
    """

    phase_1 = extract_phase(sig1)
    phase_2 = extract_phase(sig2)

    # list of coherences
    plv_list = []

    for i in range(0, len(phase_1) - window, step):
        phase_diff = phase_1[i:i+window] - phase_2[i:i+window]

        coherence = np.abs(np.mean(np.exp(1j * phase_diff)))
        plv_list.append(coherence)

    if len(plv_list) == 0:
        return {
            "mean_plv": np.nan,
            "plv_variance": np.nan
            }
    
    return {
            "mean_plv": np.mean(plv_list),
            "plv_variance": np.var(plv_list)
        }

def extract_window_features(signal, fps, window_sec=1.6, step_sec=0.8):
    """
    extract window-level features per video
    """
    # extract features per window
    WINDOW_SEC = 1.6
    STEP_SEC = 0.8

    WIN = int(WINDOW_SEC * fps)
    STEP = int(STEP_SEC * fps)

    window_features = []

    # choose reference signals
    left = signal["left_cheek"]
    right = signal["right_cheek"]
    forehead = signal["forehead"]

    for start in range(0, len(left) - WIN + 1, STEP):
        end = start + WIN

        l = left[start:end]
        r = right[start:end]
        f = forehead[start:end]

        # skip degenerate windows
        if np.std(l) == 0 or np.std(r) == 0 or np.std(f) == 0:
            continue

        # calculate heart rate
        hr = calc_heart_rate(l, fps)

        # calc correlation
        corr_lr = np.corrcoef(l, r)[0, 1]
        corr_lf = np.corrcoef(l, f)[0, 1]

        # phase coherence
        plv_lr = np.abs(np.mean(np.exp(1j * (extract_phase(l) - extract_phase(r)))))
        plv_lf = np.abs(np.mean(np.exp(1j * (extract_phase(l) - extract_phase(f)))))

        # stability
        std_l = np.std(l)
        std_r = np.std(r)
        std_f = np.std(f)

        window_features.append({
            "hr": hr,
            "corr_lr": corr_lr,
            "corr_lf": corr_lf,
            "plv_lr": plv_lr,
            "plv_lf": plv_lf,
            "std_left": std_l,
            "std_right": std_r,
            "std_forehead": std_f
        })

    return window_features

def aggregate_features(metadata, window_features, video_name):
    """
    aggregate window-level features into video features for ml prediction
    open metadata.json
    get video name and labels, append to features
    """
    if len(window_features) == 0:
        return None
    
    df = pd.DataFrame(window_features)
    video_features = {}

    for col in df.columns:
        values = df[col].dropna().values
        if len(values) == 0:
            continue
        
        video_features["video_name"] = video_name
        video_features[f"{col}_mean"] = np.mean(values)
        video_features[f"{col}_std"] = np.std(values)
        video_features[f"{col}_min"] = np.min(values)
        video_features[f"{col}_max"] = np.max(values)
        video_features[f"{col}_median"] = np.median(values)
        video_features["label"] = 1 if metadata[video_name]["label"] == "REAL" else 0

    return video_features

def process_video(metadata, video_path, video_name, output_path=None, annotate=True):
    """process a video to compute HR from face regions and optionally save annotated video."""

    video_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )

    landmarker = FaceLandmarker.create_from_options(video_options)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Cannot load video")
        return None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    timestamp_ms = 0
    frame_idx = 0
    frame_duration_ms = 1000 / fps

    out = None
    if annotate and output_path:
        out = cv2.VideoWriter(output_path,
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              fps, (frame_width, frame_height))

    # Initialize signals per region
    signals = {region: [] for region in REGIONS}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int(round(frame_idx * frame_duration_ms))
        frame_idx += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        annotated_frame = frame.copy()

        if result.face_landmarks and len(result.face_landmarks) > 0:
            face_landmarks = result.face_landmarks[0]

            # Wrap in object if list (fixes AttributeError)
            if isinstance(face_landmarks, list):
                class Temp:
                    landmark = face_landmarks
                face_landmarks = Temp()

            # Draw landmarks & regions if annotate
            if annotate:
                h, w = frame.shape[:2]
                regions = get_regions(face_landmarks, h, w)
                for points in regions.values():
                    for x, y in points:
                        cv2.circle(annotated_frame, (x, y), 2, (0,255,0), -1)

            # Extract mean RGB per region
            regions = get_regions(face_landmarks, frame_height, frame_width)
            for name, pts in regions.items():
                mean_rgb = extract_mean_values(frame, pts)
                if mean_rgb is not None:
                    signals[name].append(mean_rgb)
                else:
                    signals[name].append(signals[name][-1] if signals[name] else [0,0,0])

        # show annotated video in gui
        if annotate:
            cv2.imshow("Annotated Video", annotated_frame)
            if out:
                out.write(annotated_frame)
            if cv2.waitKey(int(1000/fps)) & 0xFF == 27:
                break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    # Process signals per region
    filtered_signals = {}
    for name, sig in signals.items():
        arr = np.array(sig)
        green = arr[:, 1]
        green = normalize_signal(green)
        green = detrend(green)
        green = bandpass_filter(green, fps)
        filtered_signals[name] = green

    # extract window-level functions
    window_features = extract_window_features(filtered_signals, fps)
    print(f"Extracted {len(window_features)} window-level feature vectors from video")

    # aggregate to video-level features
    video_features = aggregate_features(metadata, window_features, video_name)
    print("Video features have been extracted")
    
    for k, v in video_features.items():
        print(f"{k}: {v}")

    # calculate signal correlation between face regions
    region_corr = {}

    # recalc window and step for correlation, window is too small
    CORR_WINDOW = int(2.5 * fps)
    CORR_STEP = int(1.0 * fps)

    # calculate corr between left and right cheek
    region_corr["lr"] = calc_correlation(
        filtered_signals["left_cheek"],
        filtered_signals["right_cheek"],
        CORR_WINDOW,
        CORR_STEP
    )

    # calculate corr between left cheek and forehead
    region_corr["lf"] = calc_correlation(
        filtered_signals["left_cheek"],
        filtered_signals["forehead"],
        CORR_WINDOW,
        CORR_STEP
    )

    print(f"Region correlation, left-right: {region_corr['lr']} \n" 
          f"Region correlation, left-forehead: {region_corr['lf']}")
    
    print(f"Mean for left cheek signal: {np.mean(filtered_signals['left_cheek'])} \n",
          f"Mean for right cheek signal: {np.mean(filtered_signals['right_cheek'])}")
    
    # calc phase coherence
    region_phase = {}

    region_phase["lr"] = phase_coherence(
        filtered_signals["left_cheek"],
        filtered_signals["right_cheek"],
        WINDOW_SIZE,
        STEP_SIZE
    )

    region_phase["lf"] = phase_coherence(
        filtered_signals["left_cheek"],
        filtered_signals["forehead"],
        WINDOW_SIZE,
        STEP_SIZE
    )

    print("Phase coherence LR:", region_phase["lr"])
    print("Phase coherence LF:", region_phase["lf"])

    # calc heart region per face region, and their quality
    hr_per_region = {}
    quality_per_region = {}
    for name, sig in filtered_signals.items():
        segments = sliding_window_segments(sig, WINDOW_SIZE, STEP_SIZE)
        hr_windows = []
        quality_list = []
        for seg in segments:
            hr = calc_heart_rate(seg, fps)
            hr_windows.append(hr if hr is not None else np.nan)
            metrics = compute_signal_quality(seg, fps)
            metrics['hr'] = hr
            quality_list.append(metrics)
        hr_per_region[name] = [hr for hr in hr_windows if hr is not None]
        quality_per_region[name] = quality_list

    min_len = min(len(hr) for hr in hr_per_region.values() if len(hr) > 0)
    hr_matrix = np.array([hr_per_region[r][:min_len] for r in REGIONS if len(hr_per_region[r]) >= min_len])
    avg_hr = hr_matrix.mean(axis=0) if hr_matrix.size > 0 else []
    smooth_avg_hr = smooth_hr(avg_hr, window=SMOOTH_WINDOW)

    all_quality = []
    for i in range(min_len):
        row = {'window_idx': i}
        for region in REGIONS:
            if i < len(quality_per_region[region]):
                for k, v in quality_per_region[region][i].items():
                    row[f"{region}_{k}"] = v
        all_quality.append(row)
    df_quality = pd.DataFrame(all_quality)
    df_quality.to_csv('data/heart_rate_quality.csv', index=False)
    print("Saved heart rate signal quality metrics to data/hr_quality.csv")

    # plot raw vs smoothed hr
    """
        if len(avg_hr) > 0:
        plt.figure(figsize=(10,4))
        plt.plot(avg_hr, label="Raw HR", marker='o')
        plt.plot(range(len(smooth_avg_hr)), smooth_avg_hr, label="Smoothed HR", linewidth=2, marker='x')
        plt.xlabel("Window index")
        plt.ylabel("Heart Rate (bpm)")
        plt.title("Average HR across face regions")
        plt.legend()
        plt.grid(True)
        plt.show()
        print(f"Estimated average HR: {np.mean(avg_hr):.2f} bpm")
    else:
        print("No HR could be estimated")
    """
    return avg_hr, smooth_avg_hr, video_features

# run the pipeline
if __name__ == "__main__":

    with open(r"data\train_sample_videos\metadata.json", "r") as f:
        metadata = json.load(f)

    video_dir = r"data\train_sample_videos"
    video_names = list(metadata.keys())[0:1]

    all_video_features = []

    for video_name in video_names:
        video_path = os.path.join(video_dir, video_name)

        avg_hr, smooth_avg_hr, video_features = process_video(
            metadata, 
            video_path,
            video_name,
            output_path=None,
            annotate=True
        )

        if video_features is not None:
            all_video_features.append(video_features)

    df = pd.DataFrame(all_video_features)
    df.to_csv("data/video_features.csv", index=False)

    print("Saved video-level features")

