import librosa
import numpy as np

def stft(y, sr, n_fft=400, hop_t=0.010, win_t=0.025, window="hamming", 
        preemphasis=0.97):
    """
    Short time Fourier Transform
    Args:
        y(np.ndarray): raw waveform of shape (T,)
        sr(int): sample rate
        hop_t(float): spacing (in second) between consecutive frames
        win_t(float): window size (in second)
        window(str): type of window applied for STFT
        preemphasis(float): pre-emphasize raw signal with y[t] = x[t] - r*x[t-1]
    Return:
        (np.ndarray): (n_fft / 2 + 1, N) matrix; N is number of frames
    """
    if preemphasis > 1e-12:
        y = y - preemphasis * np.concatenate([[0], y[:-1]], 0)
    hop_length = int(sr * hop_t)
    win_length = int(sr * win_t)
    return librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length, 
            win_length=win_length, window=window)

def rstft(y, sr, n_fft=400, hop_t=0.010, win_t=0.025, window="hamming", 
        preemphasis=0.97, log=True, log_floor=-50):
    """
    Compute (log) magnitude spectrogram
    Args:
        y(np.ndarray): 
        sr(int):
        hop_t(float):
        win_t(float):
        window(str):
        preemphasis(float):
        log(bool):
    Return:
        (np.ndarray): (n_fft / 2 + 1, N) matrix; N is number of frames
    """
    spec = stft(y, sr, n_fft, hop_t, win_t, window, preemphasis)
    spec = np.abs(spec)
    if log:
        spec = np.log(spec)
        spec[spec < log_floor] = log_floor
    return spec

def to_melspec(y, sr, n_fft=400, hop_t=0.010, win_t=0.025, window="hamming", 
        preemphasis=0.97, n_mels=80, log=True, norm_mel=None, log_floor=-20):
    """
    Compute Mel-scale filter bank coefficients:
    Args:
        y(np.ndarray): 
        sr(int):
        hop_t(float):
        win_t(float):
        window(str):
        preemphasis(float):
        n_mels(int): number of filter banks, which are equally spaced in Mel-scale
        log(bool):
        norm_mel(None/1): normalize each filter bank to have area of 1 if set to 1;
            otherwise the peak value of eahc filter bank is 1
    Return:
        (np.ndarray): (n_mels, N) matrix; N is number of frames
    """
    spec = rstft(y, sr, n_fft, hop_t, win_t, window, preemphasis, log=False)
    hop_length = int(sr * hop_t)
    melspec = librosa.feature.melspectrogram(sr=sr, S=spec, n_fft=n_fft, 
            hop_length=hop_length, n_mels=n_mels, norm=norm_mel)
    if log:
        melspec = np.log(melspec)
        melspec[melspec < log_floor] = log_floor
    return melspec

def energy_vad(y, sr, hop_t=0.010, win_t=0.025, th_ratio=1.04/2):
    """
    Compute energy-based VAD
    """
    hop_length = int(sr * hop_t)
    win_length = int(sr * win_t)
    e = librosa.feature.rmse(y, frame_length=win_length, hop_length=hop_length)
    th = th_ratio * np.mean(e)
    vad = np.asarray(e > th, dtype=int)
    return vad
