import numpy as np
import scipy.signal
import scipy.fft
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

def load_sdr_data(data_file):
    """
    Load 16-bit quadrature measurements in the given binary file.
    """
    D = np.fromfile(data_file, dtype=np.int16)
    assert len(D) % 2 == 0

    D = D.astype(np.float32) / 2048
    i = D[::2]
    q = D[1::2]
    x = i + 1j*q

    return x


def nms(x, window_size):
    """
    Apply non-maximum suppression on x[n].

    x: x[n], input signal
    window_size: NMS window, odd-numbered
    """
    assert window_size % 2 == 1

    x_nms = np.zeros_like(x)
    half_window = window_size // 2

    for i in range(half_window, len(x) - half_window):
        xw = x[(i-half_window):(i+half_window+1)]
        if x[i] == xw.max():
            x_nms[i] = x[i]

    return x_nms

def stft(x, N, s, w, Fs):
    """
    Compute the short-time Fourier Transform using N-length FFT's with shift
    s and window w(t)

    x: x(t)
    N: FFT length (samples)
    s: window shift (samples)
    w: w(t) window function
    Fs: sampling rate

    return
    X: |DFT(x .* w)|**2 for every overlapping segment of x(t), M x N
    t: Time at the first sample of each window (s), M x 1
    """

    assert len(w) == N

    num_windows = len(range(0, len(x) - N, s))

    X = np.zeros((num_windows, N), dtype=np.float32)
    t = np.zeros(num_windows, dtype=np.float32)

    xin = np.zeros((num_windows, N), dtype=np.complex64)

    for wi, i in enumerate(range(0, len(x) - N, s)):
        xin[wi,:] = x[i:(i+N)]
        t[wi] = i / Fs

    xin *= w[None,:]
    X = scipy.fft.fft(xin, axis=1, workers=8)
    X = np.abs(X)**2

    # Offset each window time by half the window's duration
    t += (N / Fs)

    return X, t

def chirp_detection_single_pixel(x, X, pixel_band, s, Fs, t, freqs, 
                                 pixel_freq, pixel_drift_half_width,
                                 min_chirp_spacing, peak_min_snr, 
                                 debug=False):
    """
    Given the power spectral density of a single pixel's spectral band over
    time, detect the chirps.

    x and X are not required to detect chirps. They are only used for
    debug visualizations.

    x: x[n]
    X: |STFT|^2 of x[n]
    pixel_band: |STFT|^2 cropped to the pixel's spectral band
    s: window shift (samples)
    Fs: sampling rate (Hz)
    t: time for each window in the STFT
    freqs: frequency for each index in the DFT
    pixel_freq: current pixel's design frequency
    pixel_drift_half_width: pixel allowed to drif +/- this value from the design
        frequency
    chirp_detection_options: dictionary of detection options
    debug: show debugging plots
    """

    # Unpack peak detection options
    pwr = pixel_band.mean(1)

    thresh = pwr.mean() * peak_min_snr
    peak_power = pixel_band.max(1)

    # Detect a peak when the max power is above the threshold
    amplitudes_curr = peak_power.copy()
    amplitudes_curr[amplitudes_curr < thresh] = 0

    # Non-maximum suppression on the chirp power to eliminiate adjacent
    # detections
    nms_window_length = int(np.ceil(min_chirp_spacing / (s / Fs)))
    if nms_window_length % 2 == 0:
        nms_window_length += 1
    amplitudes_curr = nms(amplitudes_curr, nms_window_length)

    peak_idx_curr = np.nonzero(amplitudes_curr)
    chirp_amplitudes = amplitudes_curr[peak_idx_curr]
    chirp_times = t[peak_idx_curr]

    pixel_chirps = {
        "times": chirp_times,
        "amplitudes": chirp_amplitudes,
        "pixel_freq": pixel_freq # design frequency
    }

    if debug:
        fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(9,8))

        # Overlay detections on x(t)
        axs[0].plot((np.arange(len(x)) / Fs), np.real(x))
        axs[0].vlines(chirp_times, ymin=-1, ymax=1, colors="red", linestyles="dotted")
        axs[0].set_title("Chirp Detections")
        axs[0].set_ylim([-1, 1])
        axs[0].set_xlabel("Time (S)")
        axs[0].set_ylabel("$x(t)$")

        axs[1].plot(t, pwr)
        axs[1].set_xlabel("Time (S)")
        axs[1].set_ylabel("Power")
        axs[1].set_title("Power in Spectral Band v. Time, $f = %d$ MHz" %
                         (pixel_freq / 1e6))

        axs[2].plot(t, peak_power)
        axs[2].plot(t, np.full_like(t, thresh), 'r')
        axs[2].set_xlabel("Time (S)")
        axs[2].set_ylabel("Peak Power")
        axs[2].set_title("Peak Power in Spectral Band v. Time, $f = %d$ MHz" %
                         (pixel_freq / 1e6))

        axs[3].plot(t, peak_power / pwr.mean())
        axs[3].plot(t, np.full_like(t, peak_min_snr), 'r')
        axs[3].set_xlabel("Time (S)")
        axs[3].set_ylabel("Peak Power / Avg. Power")
        axs[3].set_title("Peak / Avg. Power in Spectral Band v. Time, $f = %d$ MHz" %
                         (pixel_freq / 1e6))

        fig.tight_layout()
        plt.show()

        fig = plt.figure()

        MAX_SUBPLOTS = 36
        # Indices in FFT
        pixel_freq_idx = np.where(
            np.abs(freqs - pixel_freq) <= pixel_drift_half_width)[0]
        for j in range(np.minimum(len(peak_idx_curr[0]), MAX_SUBPLOTS)):
            peak_psd = X[peak_idx_curr[0][j], pixel_freq_idx]

            ax = fig.add_subplot(6, 6, j+1)
            ax.plot(freqs[pixel_freq_idx] / 1e6, peak_psd)
            ax.axis("off")

        fig.suptitle("Spectrum of Chirps")
        fig.tight_layout()
        plt.show()

    return pixel_chirps

def chirp_detection(x, N, s, w, Fs, f_offset, pixel_freqs,
                   pixel_drift_half_width, chirp_detection_options,
                   debug=False):
    """
    Detect peaks in the given data

    x: x(t), 16-bit quadrature measurements, normalized to [-1, 1]
    N: fft size (samples)
    s: window shift (samples)
    w: window function
    Fs: sampling rate (Hz)
    target_freqs: list of pixel frequencies to detect (Hz)
    f_offset: center frequency (Hz)
    pixel_drift_half_width: each pixel is allowed to drif +/- this width (Hz)

    debug: plot visualizations

    returns:
    list of dictionaries providing the detections for each pixel
    """

    assert x.dtype == np.complex128 or x.dtype == np.complex64
    assert len(w) == N

    ## Compute |STFT|^2
    freqs = np.fft.fftfreq(N, d=1/Fs) + f_offset
    X, t = stft(x, N, s, w, Fs)

    if "ignore_freqs" in chirp_detection_options.keys():
        X *= ~chirp_detection_options["ignore_freqs"][None,:]

    chirp_detections = []

    min_chirp_spacing = chirp_detection_options["min_chirp_spacing"]

    # Power spectral density in each pixel's spectral band
    # (num_pixels x num_windows)
    for i, pixel_freq in enumerate(pixel_freqs):
        # Indices in FFT
        pixel_freq_idx = np.where(
            np.abs(freqs - pixel_freq) <= pixel_drift_half_width)[0]

        pixel_band = X[:,pixel_freq_idx]

        peak_min_snr = chirp_detection_options["per_pixel_threshold"][i]

        # Detect chirps within band
        chirp_detections.append(
            chirp_detection_single_pixel(x, X, pixel_band, s, Fs, t, 
                                         freqs, pixel_freq, 
                                         pixel_drift_half_width,
                                         min_chirp_spacing, 
                                         peak_min_snr, debug=debug))

    return chirp_detections


def run_example():
    N = 4096 # FFT size
    s = 2048 # window shift size
    w = scipy.signal.windows.hamming(N) # Window function
    Fs = 50e6 # Sampling rate (Hz)
    f_offset = 2055e6 # Center frequency (Hz)
    pixel_freqs = np.asarray([2050e6]) # Cricket carrier frequencies
    pixel_frequency_window_half_width = 1e6 # Cricket carrier frequencies drift within a range
    chirp_detection_options = {
        "per_pixel_threshold": [100],
        "min_chirp_spacing": 500e-6 # s
    }

    x = load_sdr_data(Path(__file__).parent / "data.bin")

    detections = chirp_detection(
        x, N, s, w, Fs, f_offset, pixel_freqs,
        pixel_frequency_window_half_width,
        chirp_detection_options, 
        debug=True)
    
    for i, freq in enumerate(pixel_freqs):
        print("Chirp detections for cricket (fc = %d MHz)" % (freq / 1e6))
        print(detections[i])
        print()


if __name__ == "__main__":
    run_example()