import numpy as np
import matplotlib.pyplot as plt

# --- System Configuration Constants ---
NUM_CHANNELS = 16          # Total channel slots in our PFB channelizer
SAMPLES_PER_SYMBOL = 4     # Oversampling factor for baseband pulse shaping
BETA = 0.25                # RRC filter roll-off factor
SPAN = 10                  # RRC filter span in symbols
NUM_SYMBOLS = 300          # Number of data symbols to stream per channel
SNR_DB = 25                # Signal-to-Noise Ratio in dB for the channel

R1 = 1.0                   # 16-APSK Inner Ring Radius
R2 = 2.75                  # 16-APSK Outer Ring Radius

# 1. 16-APSK Constellation Geometry Mapping
def get_16apsk_constellation():
    constellation = np.zeros(16, dtype=complex)
    for i in range(4):
        angle = i * (2.0 * np.pi / 4.0) + (np.pi / 4.0)
        constellation[i] = R1 * (np.cos(angle) + 1j * np.sin(angle))
    for i in range(12):
        angle = i * (2.0 * np.pi / 12.0) + (np.pi / 12.0)
        constellation[4 + i] = R2 * (np.cos(angle) + 1j * np.sin(angle))
    # Normalize average energy to 1.0
    return constellation / np.sqrt(np.mean(np.abs(constellation)**2))

# 2. Root-Raised-Cosine (RRC) Filter Generation
def rrc_filter(sps, beta, span):
    num_taps = sps * span + 1
    t = (np.arange(num_taps) - (num_taps - 1) / 2.0) / sps
    h = np.zeros(num_taps)
    for i, ti in enumerate(t):
        if ti == 0.0:
            h[i] = 1.0 - beta + (4.0 * beta / np.pi)
        elif beta != 0.0 and np.isclose(np.abs(4.0 * beta * ti), 1.0):
            h[i] = (beta / np.sqrt(2.0)) * (((1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))) + 
                    ((1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))))
        else:
            numerator = np.sin(np.pi * ti * (1.0 - beta)) + 4.0 * beta * ti * np.cos(np.pi * ti * (1.0 + beta))
            denominator = np.pi * ti * (1.0 - (4.0 * beta * ti)**2)
            h[i] = numerator / denominator
    return h / np.sqrt(np.sum(h**2))

# 3. PFB Channelizer Prototype Filter (Sinc + Hanning window)
def init_prototype_filter(num_channels, taps_per_path):
    filter_len = num_channels * taps_per_path
    t = np.arange(filter_len) - (filter_len - 1) / 2.0
    with np.errstate(divide='ignore', invalid='ignore'):
        proto_lpf = np.sin(np.pi * (1.0 / num_channels) * t) / (np.pi * (1.0 / num_channels) * t)
    proto_lpf[np.isnan(proto_lpf)] = 1.0
    window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(filter_len) / (filter_len - 1)))
    proto_lpf *= window
    proto_lpf *= (num_channels / np.sum(proto_lpf))
    return proto_lpf

# 4. Multi-Carrier Upconverter (Synthesizer via IDFT Matrix)
def multi_carrier_modulate(tx_matrix, num_channels):
    n = np.arange(num_channels).reshape(-1, 1)
    k = np.arange(num_channels).reshape(1, -1)
    transform_matrix = np.exp(1j * 2 * np.pi * k * n / num_channels)
    return np.dot(transform_matrix, tx_matrix)

# 5. Polyphase Filter Bank Channelizer Class
class PolyphaseChannelizer:
    def __init__(self, prototype_lpf, num_channels, taps_per_path):
        self.num_channels = num_channels
        self.taps_per_path = taps_per_path
        self.prototype_lpf = prototype_lpf.reshape(taps_per_path, num_channels)
        self.buffer = np.zeros((num_channels, taps_per_path), dtype=complex)
        
        # Precompute normalized DFT matrix
        k = np.arange(num_channels).reshape(-1, 1)
        n = np.arange(num_channels).reshape(1, -1)
        self.dft_matrix = np.exp(-1j * 2 * np.pi * k * n / num_channels) / num_channels

    def process_block(self, wideband_in):
        self.buffer = np.roll(self.buffer, shift=1, axis=1)
        self.buffer[:, 0] = wideband_in
        
        dft_input = np.zeros(self.num_channels, dtype=complex)
        for p in range(self.num_channels):
            dft_input[p] = np.sum(self.buffer[p, :] * self.prototype_lpf[:, p])
            
        return np.dot(self.dft_matrix, dft_input)

# --- Main Pipeline Execution ---
if __name__ == "__main__":
    np.random.seed(42)
    constellation = get_16apsk_constellation()
    h_rrc = rrc_filter(SAMPLES_PER_SYMBOL, BETA, SPAN)
    
    # Total samples per sub-channel stream
    total_samples = NUM_SYMBOLS * SAMPLES_PER_SYMBOL
    
    # Prepare matrix to hold shaped parallel streams: (16 channels, total_samples)
    tx_baseband_matrix = np.zeros((NUM_CHANNELS, total_samples), dtype=complex)
    
    # Target allocations: We will transmit data on Channel 3 and Channel 8
    active_channels = [3, 8]
    raw_symbols_saved = {}
    
    print("--- Step 1: Generating 16-APSK Baseband Signals & Pulse Shaping ---")
    for ch in active_channels:
        sym_indices = np.random.randint(0, 16, NUM_SYMBOLS)
        raw_symbols = constellation[sym_indices]
        raw_symbols_saved[ch] = raw_symbols
        
        # Create impulse train (Upsample)
        upsampled = np.zeros(total_samples, dtype=complex)
        upsampled[::SAMPLES_PER_SYMBOL] = raw_symbols
        
        # Apply RRC Filter
        tx_baseband_matrix[ch, :] = np.convolve(upsampled, h_rrc, mode='same')

    print("--- Step 2: Mixing to Multi-Carrier Wideband Stream ---")
    wideband_clean = multi_carrier_modulate(tx_baseband_matrix, NUM_CHANNELS)
    
    print("--- Step 3: Injecting AWGN Noise into Wideband Channel ---")
    # Flatten to add noise to the continuous time waveform
    flat_wideband = wideband_clean.flatten(order='F')
    sig_power = np.mean(np.abs(flat_wideband)**2)
    noise_power = sig_power / (10**(SNR_DB / 10.0))
    noise = np.sqrt(noise_power / 2.0) * (np.random.normal(0, 1, len(flat_wideband)) + 1j * np.random.normal(0, 1, len(flat_wideband)))
    wideband_noisy = flat_wideband + noise
    
    # Reshape back into blocks for the channelizer
    noisy_signal_matrix = wideband_noisy.reshape((NUM_CHANNELS, total_samples), order='F')

    print("--- Step 4: Downconverting via Polyphase Channelizer ---")
    taps_per_path = 4  # Total PFB prototype length = 16 * 4 = 64
    proto_filter = init_prototype_filter(NUM_CHANNELS, taps_per_path)
    channelizer = PolyphaseChannelizer(proto_filter, NUM_CHANNELS, taps_per_path)
    
    rx_baseband_matrix = np.zeros((NUM_CHANNELS, total_samples), dtype=complex)
    for sample_idx in range(total_samples):
        rx_baseband_matrix[:, sample_idx] = channelizer.process_block(noisy_signal_matrix[:, sample_idx])

    print("--- Step 5: Matched Filtering & Downsampling at Receiver ---")
    recovered_symbols_saved = {}
    for ch in active_channels:
        # Run through matched RRC filter to isolate from noise floor and adjacent leakage
        matched_filtered = np.convolve(rx_baseband_matrix[ch, :], h_rrc, mode='same')
        
        # Downsample back to symbol rate (account for filter delay alignment)
        delay_alignment = (SPAN * SAMPLES_PER_SYMBOL) // 2
        # Discard the initial transient settling symbols (first 15 symbols)
        recovered_symbols = matched_filtered[delay_alignment::SAMPLES_PER_SYMBOL]
        recovered_symbols_saved[ch] = recovered_symbols[15:15+NUM_SYMBOLS-30]

    # --- PLOTTING ---
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Noisy Wideband Frequency Spectrum
    plt.subplot(2, 2, 1)
    fft_vals = np.fft.fft(wideband_noisy)
    fft_freqs = np.fft.fftfreq(len(wideband_noisy))
    plt.plot(np.fft.fftshift(fft_freqs), 20 * np.log10(np.abs(np.fft.fftshift(fft_vals))), color='dimgray', alpha=0.7)
    plt.title(f"Wideband Noisy Channel Spectrum (SNR = {SNR_DB} dB)")
    plt.xlabel("Normalized Frequency")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, linestyle='--')

    # Plot 2: Time domain snapshot
    plt.subplot(2, 2, 2)
    plt.plot(wideband_noisy.real[:150], label='Real (I)', color='teal')
    plt.title("Noisy Wideband Waveform (First 150 Samples)")
    plt.xlabel("Sample Index")
    plt.grid(True, linestyle='--')
    plt.legend()

    # Plot 3: Recovered Constellation - Channel 3
    plt.subplot(2, 2, 3)
    plt.scatter(constellation.real, constellation.imag, c='red', marker='x', s=120, zorder=3, label='Ideal Ref')
    plt.scatter(recovered_symbols_saved[3].real, recovered_symbols_saved[3].imag, alpha=0.5, color='royalblue', label='Noisy RX')
    plt.title("Channel 3: Recovered 16-APSK Baseband")
    plt.axis('equal'); plt.grid(True, linestyle='--'); plt.legend()

    # Plot 4: Recovered Constellation - Channel 8
    plt.subplot(2, 2, 4)
    plt.scatter(constellation.real, constellation.imag, c='red', marker='x', s=120, zorder=3, label='Ideal Ref')
    plt.scatter(recovered_symbols_saved[8].real, recovered_symbols_saved[8].imag, alpha=0.5, color='darkgreen', label='Noisy RX')
    plt.title("Channel 8: Recovered 16-APSK Baseband")
    plt.axis('equal'); plt.grid(True, linestyle='--'); plt.legend()

    plt.tight_layout()
    plt.show()
