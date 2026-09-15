"""
Audio Processor & Test Generator Module for Gegenbauer Filter Compiler
======================================================================
Provides audio file I/O, FIR filter convolution, synthetic audio test signal generation,
and asynchronous Linux audio playback via standard system tools (aplay/ffmpeg).
"""

import os
import wave
import struct
import math
import subprocess
import threading
from typing import Tuple, Optional, List
import numpy as np
from scipy import signal


def read_wav(filepath: str) -> Tuple[int, np.ndarray]:
    """Reads a WAV file and returns (sample_rate, normalized_float_data)."""
    with wave.open(filepath, 'rb') as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    if sample_width == 2:
        dtype = np.int16
        max_val = 32768.0
    elif sample_width == 4:
        dtype = np.int32
        max_val = 2147483648.0
    elif sample_width == 1:
        dtype = np.uint8
        data = np.frombuffer(raw_bytes, dtype=dtype).astype(np.float64)
        data = (data - 128.0) / 128.0
        if n_channels > 1:
            data = data.reshape(-1, n_channels)
        return sample_rate, data
    else:
        raise ValueError(f"Unsupported sample width: {sample_width} bytes")

    data = np.frombuffer(raw_bytes, dtype=dtype).astype(np.float64) / max_val
    if n_channels > 1:
        data = data.reshape(-1, n_channels)

    return sample_rate, data


def write_wav(filepath: str, sample_rate: int, data: np.ndarray) -> str:
    """Writes float audio data (-1.0 to 1.0) to a 16-bit PCM WAV file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    clipped = np.clip(data, -1.0, 1.0)
    int16_data = (clipped * 32767.0).astype(np.int16)

    if int16_data.ndim == 1:
        n_channels = 1
    else:
        n_channels = int16_data.shape[1]

    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_data.tobytes())

    return filepath


def generate_chirp(
    duration: float = 3.0,
    sample_rate: int = 44100,
    f_start: float = 20.0,
    f_end: float = 20000.0,
    amplitude: float = 0.8
) -> Tuple[int, np.ndarray]:
    """Generates logarithmic frequency sweep (chirp) signal."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    chirp_data = amplitude * signal.chirp(t, f0=f_start, t1=duration, f1=f_end, method='logarithmic')
    return sample_rate, chirp_data


def generate_noise(
    duration: float = 3.0,
    sample_rate: int = 44100,
    noise_type: str = "white",
    amplitude: float = 0.5
) -> Tuple[int, np.ndarray]:
    """Generates synthetic noise signal (white or pink)."""
    n_samples = int(sample_rate * duration)
    if noise_type == "pink":
        white = np.random.randn(n_samples)
        X = np.fft.rfft(white)
        S = np.sqrt(1.0 / np.maximum(1.0, np.arange(len(X))))
        pink = np.fft.irfft(X * S, n=n_samples)
        peak = np.max(np.abs(pink))
        noise_data = amplitude * (pink / peak if peak > 0 else pink)
    else:
        noise_data = amplitude * (2.0 * np.random.rand(n_samples) - 1.0)
    return sample_rate, noise_data


def generate_multitone(
    duration: float = 3.0,
    sample_rate: int = 44100,
    freqs: Optional[List[float]] = None,
    amplitude: float = 0.7
) -> Tuple[int, np.ndarray]:
    """Generates multitone acoustic test signal."""
    if freqs is None:
        freqs = [220.0, 440.0, 880.0, 1760.0, 3520.0]
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sig = np.zeros_like(t)
    for f in freqs:
        sig += np.sin(2.0 * np.pi * f * t)
    peak = np.max(np.abs(sig))
    if peak > 0:
        sig = (sig / peak) * amplitude
    return sample_rate, sig


def apply_filter(taps: np.ndarray, audio_data: np.ndarray) -> np.ndarray:
    """Filters audio_data using FIR taps via FFT convolution."""
    if audio_data.ndim == 1:
        filtered = signal.fftconvolve(audio_data, taps, mode='same')
    else:
        filtered = np.zeros_like(audio_data)
        for c in range(audio_data.shape[1]):
            filtered[:, c] = signal.fftconvolve(audio_data[:, c], taps, mode='same')
    return filtered


class AudioPlayer:
    """Non-blocking audio player supporting Linux ALSA (aplay) and process management."""

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def play(self, filepath: str) -> bool:
        """Starts asynchronous playback of a WAV file."""
        self.stop()
        if not os.path.exists(filepath):
            return False

        with self._lock:
            try:
                self._process = subprocess.Popen(
                    ["aplay", "-q", filepath],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                return True
            except Exception:
                try:
                    self._process = subprocess.Popen(
                        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", filepath],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    return True
                except Exception:
                    self._process = None
                    return False

    def stop(self):
        """Stops current audio playback if active."""
        with self._lock:
            if self._process is not None:
                try:
                    self._process.terminate()
                    self._process.wait(timeout=0.2)
                except Exception:
                    try:
                        self._process.kill()
                    except Exception:
                        pass
                self._process = None

    def is_playing(self) -> bool:
        """Checks whether playback is currently active."""
        with self._lock:
            if self._process is None:
                return False
            poll = self._process.poll()
            if poll is not None:
                self._process = None
                return False
            return True
