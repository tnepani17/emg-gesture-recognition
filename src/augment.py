from scipy.signal import iirnotch, filtfilt, butter
from scipy.interpolate import CubicSpline
import numpy as np

def notch_filter(data, f0=50.0, fs=2000.0, Q=30.0):
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, data, axis=0)

def bandpass_filter(data, lowcut=20.0, highcut=450.0, fs=2000.0, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

def normalize_signal(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / (std + 1e-8)

def random_scale(signal, scale_range=(0.8, 1.2)):
    scale = np.random.uniform(*scale_range)
    return signal * scale

def time_warp(signal, warp_factor=0.1):
    orig_time = np.arange(len(signal))
    warp = np.random.normal(1, warp_factor, len(signal))
    new_time = np.cumsum(warp)
    new_time = (new_time - new_time[0]) * (orig_time[-1]/new_time[-1])
    return np.array([CubicSpline(orig_time, channel)(new_time) for channel in signal.T]).T

def prep_data(data):
    return normalize_signal(bandpass_filter(notch_filter(data)))
