import numpy as np
import matplotlib.pyplot as plt

# Configuration constants
NUM_CHANNELS = 16
TAPS_PER_PATH = 4  # Total filter length = 16 * 4 = 64
FILTER_LEN = NUM_CHANNELS * TAPS_PER_PATH
NUM_SYMBOLS = 100  # Number of symbols to stream over time

# 1. Initialize the 16-APSK Constellation mapping (DVB-S2 Standard setup)
def init_16apsk_constellation():
    r1 = 1.0   # Inner ring radius
    r2 = 2.75  # Outer ring radius
    constellation = np.zeros(16, dtype=complex)
    
    # 4 points on the inner ring
    for i in range(4):
        angle = i * (2.0 * np.pi / 4.0) + (np.pi / 4.0)
        constellation[i] = r1 * (np.cos(angle) + 1j * np.sin(angle))
        
    # 12 points on the outer ring
    for i in range(12):
        angle = i * (2.0 * np.pi / 12.0) + (np.pi / 12.0)
        constellation[4 + i] = r2 * (np.cos(angle) + 1j * np.sin(angle))
        
    return constellation

# 2. Initialize a normalized Prototype Low-Pass Filter
def init_prototype_filter():
    t = np.arange(FILTER_LEN) - (FILTER_LEN - 1) / 2.0
    with np.errstate(divide='ignore', invalid='ignore'):
        proto_lpf = np.sin(np.pi * 0.0625 * t) / (np.pi * 0.0625 * t)
    proto_lpf[np.isnan(proto_lpf)] = 1.0
    
    window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(FILTER_LEN) / (FILTER_LEN - 1)))
    proto_lpf *= window
    proto_lpf *= (NUM_CHANNELS / np.sum(proto_lpf))
    return proto_lpf

# 3. Multi-carrier Modulator (IDFT equivalent)
def multi_carrier_modulate(tx_channels_matrix):
    # tx_channels_matrix shape: (NUM_CHANNELS, NUM_SYMBOLS)
    n = np.arange(NUM_CHANNELS).reshape(-1, 1)
    k = np.arange(NUM_CHANNELS).reshape(1, -1)
    transform_matrix = np.exp(1j * 2 * np.pi * k * n / NUM_CHANNELS)
    return np.dot(transform_matrix, tx_channels_matrix)

# 4. Polyphase Filter Bank Processing Class
class PolyphaseChannelizer:
    def __init__(self, prototype_lpf):
        self.prototype_lpf = prototype_lpf.reshape(TAPS_PER_PATH, NUM_CHANNELS)
        self.buffer = np.zeros((NUM_CHANNELS, TAPS_PER_PATH), dtype=complex)

    def process_block(self, wideband_in):
        self.buffer = np.roll(self.buffer, shift=1, axis=1)
        self.buffer[:, 0] = wideband_in
        
        dft_input = np.zeros(NUM_CHANNELS, dtype=complex)
        for p in range(NUM_CHANNELS):
            dft_input[p] = np.sum(self.buffer[p, :] * self.prototype_lpf[:, p])
            
        k = np.arange(NUM_CHANNELS).reshape(-1, 1)
        n = np.arange(NUM_CHANNELS).reshape(1, -1)
        dft_matrix = np.exp(-1j * 2 * np.pi * k * n / NUM_CHANNELS) / NUM_CHANNELS
        
        return np.dot(dft_matrix, dft_input)

# --- Execution ---
if __name__ == "__main__":
    apsk_constellation = init_16apsk_constellation()
    prototype_filter = init_prototype_filter()
    
    # Generate random streams of 16-APSK indices for Channel 2 and Channel 5
    np.random.seed(42) # For reproducible plots
    ch2_indices = np.random.randint(0, 16, NUM_SYMBOLS)
    ch5_indices = np.random.randint(0, 16, NUM_SYMBOLS)
    
    # Map indices to complex symbols
    tx_stream = np.zeros((NUM_CHANNELS, NUM_SYMBOLS), dtype=complex)
    tx_stream[2, :] = apsk_constellation[ch2_indices]
    tx_stream[5, :] = apsk_constellation[ch5_indices]
    
    # Synthesize the wideband signal over time
    # This creates a matrix of size (16 channels, 100 symbol intervals)
    wideband_signal = multi_carrier_modulate(tx_stream)
    
    # Flatten the wideband signal into a continuous time-series stream for analysis
    wideband_time_series = wideband_signal.flatten(order='F')
    
    # Initialize the channelizer state machine
    channelizer = PolyphaseChannelizer(prototype_filter)
    
    # Process the stream block-by-block
    rx_stream = np.zeros((NUM_CHANNELS, NUM_SYMBOLS), dtype=complex)
    for sym in range(NUM_SYMBOLS):
        rx_stream[:, sym] = channelizer.process_block(wideband_signal[:, sym])
        
    # --- PLOTTING ---
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Combined Wideband Signal (Time Domain)
    plt.subplot(2, 2, 1)
    plt.plot(wideband_time_series.real[:200], label='Real (I)', color='teal', alpha=0.8)
    plt.plot(wideband_time_series.imag[:200], label='Imag (Q)', color='orange', alpha=0.6)
    plt.title("Wideband Combined Signal (First 200 Samples)")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle='--')
    plt.legend()
    
    # Plot 2: Frequency Spectrum of Wideband Signal
    plt.subplot(2, 2, 2)
    fft_vals = np.fft.fft(wideband_time_series)
    fft_freqs = np.fft.fftfreq(len(wideband_time_series))
    plt.plot(np.fft.fftshift(fft_freqs), 20 * np.log10(np.abs(np.fft.fftshift(fft_vals))), color='purple')
    plt.title("Wideband Signal Spectrum (Showing Carrier Peaks)")
    plt.xlabel("Normalized Frequency")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, linestyle='--')
    
    # Discard the first 4 transient settling symbols for cleaner constellation tracking
    settled_rx = rx_stream[:, 4:] 
    
    # Plot 3: Recovered Constellation - Channel 2
    plt.subplot(2, 2, 3)
    plt.scatter(apsk_constellation.real, apsk_constellation.imag, c='red', marker='x', s=100, label='Ideal Reference')
    plt.scatter(settled_rx[2, :].real, settled_rx[2, :].imag, alpha=0.6, color='blue', edgecolors='none', label='Recovered RX')
    plt.title("Channel 2: Recovered 16-APSK Constellation")
    plt.xlabel("In-Phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.axis('equal')
    plt.grid(True, linestyle='--')
    plt.legend()
    
    # Plot 4: Recovered Constellation - Channel 5
    plt.subplot(2, 2, 4)
    plt.scatter(apsk_constellation.real, apsk_constellation.imag, c='red', marker='x', s=100, label='Ideal Reference')
    plt.scatter(settled_rx[5, :].real, settled_rx[5, :].imag, alpha=0.6, color='green', edgecolors='none', label='Recovered RX')
    plt.title("Channel 5: Recovered 16-APSK Constellation")
    plt.xlabel("In-Phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.axis('equal')
    plt.grid(True, linestyle='--')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
