import numpy as np
import matplotlib.pyplot as plt

# Configuration constants
NUM_CHANNELS = 16
TAPS_PER_PATH = 4  
FILTER_LEN = NUM_CHANNELS * TAPS_PER_PATH
NUM_SYMBOLS = 200  # Increased symbol count to see the noise distribution clearly

# 1. Initialize the 16-APSK Constellation mapping
def init_16apsk_constellation():
    r1 = 1.0   
    r2 = 2.75  
    constellation = np.zeros(16, dtype=complex)
    for i in range(4):
        angle = i * (2.0 * np.pi / 4.0) + (np.pi / 4.0)
        constellation[i] = r1 * (np.cos(angle) + 1j * np.sin(angle))
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

# 3. Multi-carrier Modulator
def multi_carrier_modulate(tx_channels_matrix):
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

if __name__ == "__main__":
    np.random.seed(42)
    apsk_constellation = init_16apsk_constellation()
    prototype_filter = init_prototype_filter()
    
    # Generate random streams of 16-APSK symbols for Channel 2 and Channel 5
    ch2_indices = np.random.randint(0, 16, NUM_SYMBOLS)
    ch5_indices = np.random.randint(0, 16, NUM_SYMBOLS)
    
    tx_stream = np.zeros((NUM_CHANNELS, NUM_SYMBOLS), dtype=complex)
    tx_stream[2, :] = apsk_constellation[ch2_indices]
    tx_stream[5, :] = apsk_constellation[ch5_indices]
    
    # 1. Synthesize clean wideband signal
    wideband_signal = multi_carrier_modulate(tx_stream)
    
    # 2. INTRODUCE IMPAIRMENTS TO THE WIDEBAND TIME STREAM
    # A. Multipath Fading: Convolve the wideband signal with a basic 2-tap channel filter
    # This distorts the frequency amplitude and phase profile across subchannels.
    channel_impulse = np.array([1.0, 0.4 + 0.31j]) 
    flat_wideband = wideband_signal.flatten(order='F')
    faded_wideband = np.convolve(flat_wideband, channel_impulse, mode='same')
    
    # B. Phase Noise: Simulate oscillator phase jitter tracking over time
    phase_jitter_std = 0.05  # Radians
    phase_noise = np.exp(1j * np.random.normal(0, phase_jitter_std, len(faded_wideband)))
    corrupted_wideband = faded_wideband * phase_noise
    
    # Reshape back into parallel structural blocks for the PFB channelizer
    corrupted_signal_matrix = corrupted_wideband.reshape((NUM_CHANNELS, NUM_SYMBOLS), order='F')
    
    # 3. CHANNELIZATION AT RECEIVER
    channelizer = PolyphaseChannelizer(prototype_filter)
    rx_stream_corrupted = np.zeros((NUM_CHANNELS, NUM_SYMBOLS), dtype=complex)
    for sym in range(NUM_SYMBOLS):
        rx_stream_corrupted[:, sym] = channelizer.process_block(corrupted_signal_matrix[:, sym])
        
    # Discard transient settling blocks
    settled_corrupted = rx_stream_corrupted[:, 4:]
    
    # 4. ONE-TAP EQUALIZATION (COMPENSATION)
    # Re-calculate the frequency channel frequency response of our multipath impulse
    channel_fft = np.fft.fft(channel_impulse, NUM_CHANNELS)
    
    # Equalize Channel 2 and Channel 5 via a simple zero-forcing division step
    equalized_ch2 = settled_corrupted[2, :] / channel_fft[2]
    equalized_ch5 = settled_corrupted[5, :] / channel_fft[5]

    # --- PLOTTING ---
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Channel 2 - Distorted by Phase Noise & Fading
    plt.subplot(2, 2, 1)
    plt.scatter(apsk_constellation.real, apsk_constellation.imag, c='red', marker='x', s=100, label='Ideal Ref')
    plt.scatter(settled_corrupted[2, :].real, settled_corrupted[2, :].imag, alpha=0.6, color='brown', label='Corrupted RX')
    plt.title("Channel 2: Distorted Constellation\n(Phase Noise & Fading Effects)")
    plt.xlabel("I"); plt.ylabel("Q"); plt.axis('equal'); plt.grid(True, linestyle='--'); plt.legend()
    
    # Plot 2: Channel 2 - Restored via Equaliser
    plt.subplot(2, 2, 2)
    plt.scatter(apsk_constellation.real, apsk_constellation.imag, c='red', marker='x', s=100, label='Ideal Ref')
    plt.scatter(equalized_ch2.real, equalized_ch2.imag, alpha=0.6, color='blue', label='Equalized RX')
    plt.title("Channel 2: Recovered Constellation\n(After One-Tap Equalisation)")
    plt.xlabel("I"); plt.ylabel("Q"); plt.axis('equal'); plt.grid(True, linestyle='--'); plt.legend()
    
    # Plot 3: Channel 5 - Distorted by Phase Noise & Fading
    plt.subplot(2, 2, 3)
    plt.scatter(apsk_constellation.real, apsk_constellation.imag, c='red', marker='x', s=100, label='Ideal Ref')
    plt.scatter(settled_corrupted[5, :].real, settled_corrupted[5, :].imag, alpha=0.6, color='brown', label='Corrupted RX')
    plt.title("Channel 5: Distorted Constellation\n(Phase Noise & Fading Effects)")
    plt.xlabel("I"); plt.ylabel("Q"); plt.axis('equal'); plt.grid(True, linestyle='--'); plt.legend()
    
    # Plot 4: Channel 5 - Restored via Equaliser
    plt.subplot(2, 2, 4)
    plt.scatter(apsk_constellation.real, apsk_constellation.imag, c='red', marker='x', s=100, label='Ideal Ref')
    plt.scatter(equalized_ch5.real, equalized_ch5.imag, alpha=0.6, color='green', label='Equalized RX')
    plt.title("Channel 5: Recovered Constellation\n(After One-Tap Equalisation)")
    plt.xlabel("I"); plt.ylabel("Q"); plt.axis('equal'); plt.grid(True, linestyle='--'); plt.legend()
    
    plt.tight_layout()
    plt.show()
