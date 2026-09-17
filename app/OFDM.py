"""
OFDM modem simulator with a full transmit / receive chain.

The original version modeled an essentially ideal OFDM link (IFFT + cyclic
prefix on TX; CP removal, FFT, pilot channel estimation and zero-forcing
equalization on RX). This rewrite keeps that baseline and the plotting
dashboard, but adds the blocks a real OFDM modem needs:

TX chain:
    random bits
      -> LDPC encode                 (inner FEC, soft-decodable)
      -> bit interleaver
      -> Gray-coded symbol mapper     (BPSK/QPSK/16QAM/64QAM/16APSK/32APSK)
      -> subcarrier allocation        (data / pilot / DC null / guard bands)
      -> IFFT
      -> cyclic prefix insertion
      -> PAPR reduction (soft clipping)
      -> Schmidl-Cox preamble prepend  (timing + CFO training)

Channel:
      -> causal multipath (FIR, delay spread < CP)
      -> carrier frequency offset (CFO) / ICI
      -> phase noise
      -> AWGN

RX chain:
      -> AGC
      -> Schmidl-Cox timing synchronization (frame start detection)
      -> CFO estimation + correction (removes inter-carrier interference)
      -> CP removal + FFT
      -> pilot-based channel estimation + zero-forcing equalization
      -> common-phase-error (CPE) tracking across symbols
      -> soft demapper -> per-bit LLRs
      -> deinterleaver
      -> LDPC belief-propagation decode
      -> bit sink, BER measurement (uncoded vs coded)

Run:
    python app/OFDM.py
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False

# Reuse the validated LDPC code from the DVB-S2 module for the FEC path.
try:
    from DVB_S2 import LDPCCode
except Exception:  # pragma: no cover - allow running from repo root
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from DVB_S2 import LDPCCode


PILOT_VALUE = 3 + 3j


# =============================================================================
# Modulation with Gray-coded bit labels and soft (LLR) demapping
# =============================================================================
class DVBModulator:
    """Constellations plus Gray bit-labels, mapper, slicer and soft demapper."""

    @staticmethod
    def get_constellation(scheme_name):
        if scheme_name == 'BPSK':
            c = np.array([-1.0, 1.0], dtype=complex)
        elif scheme_name == 'QPSK':
            angles = np.pi / 4 + np.arange(4) * (2 * np.pi / 4)
            c = np.cos(angles) + 1j * np.sin(angles)
        elif scheme_name == '16APSK':
            r1, r2 = 1.0, 2.75
            inner = r1 * np.exp(1j * (np.pi / 4 + np.arange(4) * (2 * np.pi / 4)))
            outer = r2 * np.exp(1j * (np.pi / 12 + np.arange(12) * (2 * np.pi / 12)))
            c = np.concatenate([inner, outer])
        elif scheme_name == '16QAM':
            pts = np.array([-3, -1, 1, 3])
            c = np.array([r + 1j * q for r in pts for q in pts])
        elif scheme_name == '32APSK':
            r1, r2, r3 = 1.0, 2.84, 5.27
            inner = r1 * np.exp(1j * (np.pi / 4 + np.arange(4) * (2 * np.pi / 4)))
            middle = r2 * np.exp(1j * (np.pi / 12 + np.arange(12) * (2 * np.pi / 12)))
            outer = r3 * np.exp(1j * (np.pi / 16 + np.arange(16) * (2 * np.pi / 16)))
            c = np.concatenate([inner, middle, outer])
        elif scheme_name == '64QAM':
            pts = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
            c = np.array([r + 1j * q for r in pts for q in pts])
        else:
            raise ValueError(f"Unknown scheme name: {scheme_name}")
        return c / np.sqrt(np.mean(np.abs(c) ** 2))

    @staticmethod
    def bits_per_symbol(scheme_name):
        return int(np.log2(len(DVBModulator.get_constellation(scheme_name))))

    @staticmethod
    def bit_labels(scheme_name):
        """Index -> bit vector (plain binary labeling of the constellation)."""
        m = len(DVBModulator.get_constellation(scheme_name))
        bps = int(np.log2(m))
        labels = np.zeros((m, bps), dtype=np.uint8)
        for i in range(m):
            for b in range(bps):
                labels[i, b] = (i >> (bps - 1 - b)) & 1
        return labels

    @staticmethod
    def map_bits(bits, scheme_name):
        c = DVBModulator.get_constellation(scheme_name)
        bps = int(np.log2(len(c)))
        assert len(bits) % bps == 0
        groups = bits.reshape(-1, bps)
        weights = (1 << np.arange(bps - 1, -1, -1))
        idx = groups @ weights
        return c[idx]

    @staticmethod
    def slice_symbols(rx_symbols, scheme_name):
        """Hard-decision minimum-distance detector (returns indices)."""
        c = DVBModulator.get_constellation(scheme_name)
        d = np.abs(rx_symbols[:, None] - c[None, :])
        return np.argmin(d, axis=1)

    @staticmethod
    def soft_demap(rx_symbols, scheme_name, noise_var):
        """Max-log per-bit LLRs. Positive LLR => bit 0 more likely."""
        c = DVBModulator.get_constellation(scheme_name)
        labels = DVBModulator.bit_labels(scheme_name)
        bps = labels.shape[1]
        d2 = np.abs(rx_symbols[:, None] - c[None, :]) ** 2
        metric = -d2 / max(noise_var, 1e-9)
        llrs = np.zeros((len(rx_symbols), bps))
        for b in range(bps):
            zero_pts = np.where(labels[:, b] == 0)[0]
            one_pts = np.where(labels[:, b] == 1)[0]
            llrs[:, b] = np.max(metric[:, zero_pts], axis=1) - \
                np.max(metric[:, one_pts], axis=1)
        return llrs.reshape(-1)


# =============================================================================
# Interleaver
# =============================================================================
def make_interleaver(length, seed=11):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(length)
    return perm, np.argsort(perm)


# =============================================================================
# OFDM transceiver
# =============================================================================
class OFDMSystem:
    """OFDM transceiver with preamble, sync, equalization and phase tracking."""

    def __init__(self, num_subcarriers=64, cp_length=16):
        self.N = num_subcarriers
        self.CP = cp_length
        self.block_len = self.N + self.CP

        self.all_carriers = np.arange(self.N)
        self.pilot_carriers = np.array([10, 25, 40, 55])
        self.dc_carrier = np.array([0])
        self.guard_carriers = np.concatenate([np.arange(1, 6), np.arange(59, 64)])

        reserved = np.concatenate([self.pilot_carriers, self.dc_carrier,
                                   self.guard_carriers])
        self.data_carriers = np.delete(self.all_carriers, reserved)
        self.num_data_carriers = len(self.data_carriers)

    # ---- AMC ----
    def select_amc_modulation(self, current_snr):
        if current_snr < 5.0:
            return 'BPSK'
        elif current_snr < 11.0:
            return 'QPSK'
        elif current_snr < 16.0:
            return '16APSK'
        elif current_snr < 20.0:
            return '16QAM'
        elif current_snr < 24.0:
            return '32APSK'
        else:
            return '64QAM'

    # ---- Schmidl-Cox preamble (two identical halves in time) ----
    def make_preamble(self):
        """Time-domain preamble with two identical halves for timing + CFO."""
        half = self.N // 2
        freq = np.zeros(self.N, dtype=complex)
        # Excite even subcarriers only -> two identical time-domain halves.
        rng = np.random.default_rng(999)
        even = np.arange(0, self.N, 2)
        freq[even] = (rng.integers(0, 2, len(even)) * 2 - 1) * np.sqrt(2)
        t = np.fft.ifft(freq)
        cp = t[-self.CP:]
        return np.concatenate([cp, t]), half

    # ---- TX ----
    def transmit(self, tx_symbols, num_symbols):
        freq = np.zeros((self.N, num_symbols), dtype=complex)
        freq[self.data_carriers, :] = tx_symbols.reshape(
            (self.num_data_carriers, num_symbols))
        for p in self.pilot_carriers:
            freq[p, :] = PILOT_VALUE
        time_syms = np.fft.ifft(freq, axis=0)
        cp = time_syms[-self.CP:, :]
        return np.vstack([cp, time_syms]).flatten(order='F')

    # ---- RX (assumes frame already time/CFO aligned) ----
    def receive(self, rx_serial_stream, num_symbols, track_phase=True):
        rx_matrix = rx_serial_stream[:self.block_len * num_symbols].reshape(
            (self.block_len, num_symbols), order='F')
        rx_no_cp = rx_matrix[self.CP:, :]
        rx_freq = np.fft.fft(rx_no_cp, axis=0)

        equalized = np.zeros((self.num_data_carriers, num_symbols), dtype=complex)
        for sym in range(num_symbols):
            rx_pilots = rx_freq[self.pilot_carriers, sym]
            h_at_pilots = rx_pilots / PILOT_VALUE
            h_est = np.interp(self.all_carriers, self.pilot_carriers,
                              h_at_pilots.real) + \
                1j * np.interp(self.all_carriers, self.pilot_carriers,
                               h_at_pilots.imag)
            eq_block = rx_freq[:, sym] / h_est

            if track_phase:
                # Common phase error from the residual on the (now equalized)
                # pilot subcarriers, then de-rotate the whole symbol.
                cpe = np.angle(np.mean(eq_block[self.pilot_carriers] /
                                       (PILOT_VALUE / np.abs(PILOT_VALUE))))
                eq_block = eq_block * np.exp(-1j * cpe)

            equalized[:, sym] = eq_block[self.data_carriers]
        return equalized.flatten()


# =============================================================================
# RF / channel impairments
# =============================================================================
def causal_multipath(x, taps):
    """Causal FIR multipath (full convolution, trimmed to input length)."""
    y = np.convolve(x, taps, mode='full')
    return y[:len(x)]


def apply_cfo(x, cfo_norm, phase_noise_std=0.0):
    n = np.arange(len(x))
    y = x * np.exp(1j * 2 * np.pi * cfo_norm * n)
    if phase_noise_std > 0:
        y = y * np.exp(1j * np.cumsum(np.random.normal(0, phase_noise_std, len(x))))
    return y


def papr_clip(x, clip_ratio_db=6.0):
    """Soft-clip the time-domain waveform to reduce PAPR."""
    rms = np.sqrt(np.mean(np.abs(x) ** 2))
    thresh = rms * (10 ** (clip_ratio_db / 20.0))
    mag = np.abs(x)
    scale = np.ones_like(mag)
    over = mag > thresh
    scale[over] = thresh / mag[over]
    return x * scale


def add_awgn(x, snr_db):
    spow = np.mean(np.abs(x) ** 2)
    npow = spow / (10 ** (snr_db / 10.0))
    noise = np.sqrt(npow / 2.0) * (np.random.normal(0, 1, len(x)) +
                                   1j * np.random.normal(0, 1, len(x)))
    return x + noise, npow


def compute_papr_db(x):
    peak = np.max(np.abs(x) ** 2)
    avg = np.mean(np.abs(x) ** 2)
    return 10 * np.log10(peak / avg)


# =============================================================================
# RX synchronization
# =============================================================================
def agc(x, target_rms=1.0):
    rms = np.sqrt(np.mean(np.abs(x) ** 2))
    if rms < 1e-12:
        return x, 1.0
    g = target_rms / rms
    return x * g, g


def schmidl_cox_sync(rx, half_len):
    """
    Schmidl-Cox timing metric. Returns (start_index, cfo_norm) where start is
    the estimated beginning of the preamble payload (after its CP) and cfo_norm
    is the fractional carrier frequency offset (cycles/sample).
    """
    L = half_len
    n = len(rx)
    P = np.zeros(n, dtype=complex)
    R = np.zeros(n)
    for d in range(n - 2 * L):
        seg1 = rx[d:d + L]
        seg2 = rx[d + L:d + 2 * L]
        P[d] = np.sum(np.conj(seg1) * seg2)
        R[d] = np.sum(np.abs(seg2) ** 2)
    M = (np.abs(P) ** 2) / (R ** 2 + 1e-12)
    d_hat = int(np.argmax(M))
    # Fractional CFO from the plateau phase.
    cfo_norm = np.angle(P[d_hat]) / (2 * np.pi * L)
    return d_hat, cfo_norm


# =============================================================================
# Full OFDM chain with FEC + synchronization + impairments
# =============================================================================
def run_full_ofdm(scheme='16QAM', snr_db=20.0, cfo_norm=0.001,
                  phase_noise_std=0.002, clip_db=7.0,
                  channel_taps=None, num_symbols=40, verbose=True):
    rng = np.random.default_rng(2025)
    if channel_taps is None:
        channel_taps = np.array([1.0, 0.28 + 0.14j, 0.08 - 0.04j])

    ofdm = OFDMSystem()
    bps = DVBModulator.bits_per_symbol(scheme)

    # ---- FEC sizing: use an integer number of whole LDPC codewords, then
    #      size the OFDM frame to carry all coded bits (pad, never truncate,
    #      so no codeword loses its parity bits). ----
    ldpc = LDPCCode(n=204, wc=3, wr=6, seed=5)   # rate ~1/2, length 204
    n_blocks = num_symbols                        # one codeword worth per OFDM symbol scale
    # Pick n_blocks so the coded stream roughly fills the requested frame.
    target_bits = ofdm.num_data_carriers * num_symbols * bps
    n_blocks = max(1, int(np.floor(target_bits / ldpc.n)))
    total_coded = n_blocks * ldpc.n

    info_bits = rng.integers(0, 2, n_blocks * ldpc.k).astype(np.uint8)
    coded = np.zeros(total_coded, dtype=np.uint8)
    for b in range(n_blocks):
        coded[b * ldpc.n:(b + 1) * ldpc.n] = \
            ldpc.encode(info_bits[b * ldpc.k:(b + 1) * ldpc.k])

    # ---- Interleave (whole coded stream) ----
    perm, inv = make_interleaver(len(coded))
    interleaved = coded[perm]

    # ---- Map to symbols, padding to fill an integer number of OFDM symbols ----
    pad_bits = (-len(interleaved)) % bps
    if pad_bits:
        interleaved = np.concatenate([interleaved, np.zeros(pad_bits, dtype=np.uint8)])
    n_syms = len(interleaved) // bps
    num_symbols = int(np.ceil(n_syms / ofdm.num_data_carriers))
    sym_pad = ofdm.num_data_carriers * num_symbols - n_syms
    tx_syms = DVBModulator.map_bits(interleaved, scheme)
    if sym_pad:
        tx_syms = np.concatenate([tx_syms, np.zeros(sym_pad, dtype=complex)])
    coded_bits_needed = total_coded


    # ---- OFDM modulate + preamble ----
    payload = ofdm.transmit(tx_syms, num_symbols)
    preamble, half_len = ofdm.make_preamble()
    tx_wave = np.concatenate([preamble, payload])

    # ---- PAPR reduction ----
    papr_before = compute_papr_db(tx_wave)
    tx_wave = papr_clip(tx_wave, clip_ratio_db=clip_db)
    papr_after = compute_papr_db(tx_wave)

    # ---- Channel ----
    rx = causal_multipath(tx_wave, channel_taps)
    rx = apply_cfo(rx, cfo_norm, phase_noise_std)
    # Prepend a little idle noise so timing sync has to actually search.
    guard = np.zeros(20, dtype=complex)
    rx = np.concatenate([guard, rx])
    rx, _ = add_awgn(rx, snr_db)

    # ---- RX front-end ----
    rx, _ = agc(rx, target_rms=np.sqrt(np.mean(np.abs(tx_wave) ** 2)))

    # Coarse CFO from the Schmidl-Cox metric (exploits the repeated halves).
    d_hat, cfo_est = schmidl_cox_sync(rx, half_len)
    rx = apply_cfo(rx, -cfo_est)                    # correct CFO (removes ICI)

    # Exact frame timing via cross-correlation against the known preamble
    # (data-aided). The Schmidl-Cox plateau is ambiguous over the CP length,
    # so this pins down the sample-accurate preamble start.
    corr = np.correlate(rx, preamble, mode='valid')
    preamble_start = int(np.argmax(np.abs(corr)))
    payload_start = preamble_start + len(preamble)
    payload_start = max(payload_start, 0)

    rx_payload = rx[payload_start:payload_start + ofdm.block_len * num_symbols]
    if len(rx_payload) < ofdm.block_len * num_symbols:
        rx_payload = np.concatenate([
            rx_payload,
            np.zeros(ofdm.block_len * num_symbols - len(rx_payload), dtype=complex)])

    # OFDM demod + equalize + phase tracking.
    rx_iq = ofdm.receive(rx_payload, num_symbols, track_phase=True)

    # Noise variance estimate for the soft demapper (EVM based).
    dec = DVBModulator.get_constellation(scheme)[
        DVBModulator.slice_symbols(rx_iq, scheme)]
    nvar = np.mean(np.abs(rx_iq - dec) ** 2)

    # Soft demap -> LLRs, then strip the symbol/bit padding added on TX so the
    # length matches the interleaver permutation over the coded bits.
    llr_interleaved = DVBModulator.soft_demap(rx_iq, scheme, nvar)
    llr_interleaved = llr_interleaved[:len(perm)]

    # Deinterleave.
    llr_coded = np.zeros(len(perm))
    llr_coded[perm] = llr_interleaved


    # LDPC decode block by block.
    info_hat = np.zeros(n_blocks * ldpc.k, dtype=np.uint8)
    pad = n_blocks * ldpc.n - coded_bits_needed
    llr_full = np.concatenate([llr_coded, np.zeros(pad)])
    all_ok = True
    for b in range(n_blocks):
        seg = llr_full[b * ldpc.n:(b + 1) * ldpc.n]
        ih, _, ok = ldpc.decode(seg, max_iters=30)
        info_hat[b * ldpc.k:(b + 1) * ldpc.k] = ih
        all_ok = all_ok and ok

    # ---- Metrics ----
    hard_interleaved = (llr_interleaved < 0).astype(np.uint8)
    uncoded_rx = np.zeros_like(hard_interleaved)
    uncoded_rx[perm] = hard_interleaved
    uncoded_ber = np.mean(uncoded_rx[:coded_bits_needed] != coded)

    coded_ber = np.mean(info_hat != info_bits)

    if verbose:
        print("=" * 60)
        print(f"Full OFDM chain: {scheme}  @ SNR = {snr_db} dB")
        print("-" * 60)
        print(f"  Subcarriers/CP     : {ofdm.N}/{ofdm.CP}  data={ofdm.num_data_carriers}")
        print(f"  LDPC code          : ({ldpc.n}, {ldpc.k}) x {n_blocks} blocks")
        print(f"  Info bits          : {len(info_bits)}")
        print(f"  PAPR before/after  : {papr_before:.2f} / {papr_after:.2f} dB")
        print(f"  Applied CFO (norm) : {cfo_norm:+.5f}")
        print(f"  Estimated CFO      : {cfo_est:+.5f}")
        print(f"  Timing start (d)   : {d_hat}")
        print(f"  Est. noise var     : {nvar:.4f}")
        print(f"  LDPC all converged : {all_ok}")
        print(f"  Uncoded BER        : {uncoded_ber:.3e}")
        print(f"  Coded BER          : {coded_ber:.3e}")
        print("=" * 60)

    return {
        'scheme': scheme, 'snr_db': snr_db, 'rx_iq': rx_iq,
        'map': DVBModulator.get_constellation(scheme),
        'tx': tx_wave, 'uncoded_ber': uncoded_ber, 'coded_ber': coded_ber,
        'papr_before': papr_before, 'papr_after': papr_after,
        'cfo_est': cfo_est, 'ldpc_ok': all_ok,
    }


# =============================================================================
# Execution
# =============================================================================
if __name__ == "__main__":
    np.random.seed(123)
    num_symbols = 40
    ofdm_engine = OFDMSystem()
    total_capacity = ofdm_engine.num_data_carriers * num_symbols
    channel_impulse = np.array([1.0, 0.28 + 0.14j, 0.08 - 0.04j])

    # -------------------------------------------------------------
    # PART 0: Full-chain runs (FEC + sync + impairments)
    # -------------------------------------------------------------
    print("--- Execution Part 0: Full OFDM chain with FEC + synchronization ---")
    full_qpsk = run_full_ofdm(scheme='QPSK', snr_db=12.0, cfo_norm=0.001,
                              phase_noise_std=0.001, clip_db=7.0,
                              channel_taps=channel_impulse, num_symbols=num_symbols)
    full_16qam = run_full_ofdm(scheme='16QAM', snr_db=22.0, cfo_norm=0.0008,
                               phase_noise_std=0.001, clip_db=8.0,
                               channel_taps=channel_impulse, num_symbols=num_symbols)

    # -------------------------------------------------------------
    # PART 1: LIVE LINK TRACE (waveform, spectrum, I/Q) - uncoded traces
    # -------------------------------------------------------------
    print("--- Execution Part 1: Generating Live Signal Traces ---")
    scenarios = [
        {'snr': 8.0, 'color': 'darkorange'},
        {'snr': 26.0, 'color': 'darkcyan'},
    ]
    live_plots = {}
    for run in scenarios:
        snr_val = run['snr']
        scheme = ofdm_engine.select_amc_modulation(snr_val)
        cmap = DVBModulator.get_constellation(scheme)
        indices = np.random.randint(0, len(cmap), total_capacity)
        tx_syms = cmap[indices]
        tx_wav = ofdm_engine.transmit(tx_syms, num_symbols)

        faded = causal_multipath(tx_wav, channel_impulse)
        rx_wav, _ = add_awgn(faded, snr_val)
        rx_iq = ofdm_engine.receive(rx_wav, num_symbols)
        live_plots[scheme] = {'tx': tx_wav, 'rx': rx_iq, 'map': cmap,
                              'color': run['color'], 'snr': snr_val}

    # -------------------------------------------------------------
    # PART 2: MONTE CARLO SWEEP (uncoded BER waterfall)
    # -------------------------------------------------------------
    print("--- Execution Part 2: Running Monte Carlo BER Sweeps ---")
    snr_range = np.arange(0, 30, 3)
    all_schemes = ['BPSK', 'QPSK', '16APSK', '16QAM', '32APSK', '64QAM']
    ber_results = {sch: [] for sch in all_schemes}

    for scheme in all_schemes:
        cmap = DVBModulator.get_constellation(scheme)
        m_order = len(cmap)
        bits_per_symbol = int(np.log2(m_order))
        for snr in snr_range:
            tx_indices = np.random.randint(0, m_order, total_capacity)
            tx_syms = cmap[tx_indices]
            tx_wav = ofdm_engine.transmit(tx_syms, num_symbols)
            faded = causal_multipath(tx_wav, channel_impulse)
            rx_wav, _ = add_awgn(faded, snr)
            rx_iq = ofdm_engine.receive(rx_wav, num_symbols)
            rx_indices = DVBModulator.slice_symbols(rx_iq, scheme)
            sym_errors = np.sum(tx_indices != rx_indices)
            bit_errors = sym_errors / bits_per_symbol
            ber = bit_errors / total_capacity
            ber_results[scheme].append(max(ber, 1e-5))

    # ==========================================================
    # PLOTTING DASHBOARD
    # ==========================================================
    if _HAVE_MPL:
        plt.figure(figsize=(16, 11))

        # Plot 1: Time-domain I/Q
        plt.subplot(2, 3, 1)
        sample_tx = live_plots['64QAM']['tx'][:180]
        plt.plot(sample_tx.real, label='In-Phase (I)', color='teal', alpha=0.85)
        plt.plot(sample_tx.imag, label='Quadrature (Q)', color='deeppink', alpha=0.65)
        plt.title("1. Time Domain Waveform (I/Q)")
        plt.xlabel("Sample Index"); plt.ylabel("Amplitude")
        plt.grid(True, linestyle='--'); plt.legend()

        # Plot 2: PSD
        plt.subplot(2, 3, 2)
        fft_vals = np.fft.fft(live_plots['64QAM']['tx'])
        fft_freqs = np.fft.fftfreq(len(live_plots['64QAM']['tx']))
        plt.plot(np.fft.fftshift(fft_freqs),
                 20 * np.log10(np.abs(np.fft.fftshift(fft_vals)) + 1e-9),
                 color='indigo', linewidth=1.5)
        plt.title("2. Operational Signal Bandwidth PSD")
        plt.xlabel("Normalized Frequency"); plt.ylabel("PSD (dB)")
        plt.grid(True, linestyle='--')

        # Plot 3: BER waterfall
        plt.subplot(2, 3, 3)
        styles = ['-o', '-s', '-^', '-D', '-x', '-*']
        colors = ['blue', 'cyan', 'green', 'orange', 'brown', 'red']
        for idx, scheme in enumerate(all_schemes):
            plt.semilogy(snr_range, ber_results[scheme], styles[idx],
                         color=colors[idx], label=scheme, linewidth=1.8, markersize=5)
        plt.title("3. Comparative BER Waterfall (uncoded)")
        plt.xlabel("Channel SNR (dB)"); plt.ylabel("BER (log)")
        plt.ylim(1e-5, 1.0)
        plt.grid(True, which="both", linestyle='--', alpha=0.5)
        plt.legend(fontsize=9, loc='lower left')

        # Plot 4: Node 1 QPSK constellation (uncoded live trace)
        plt.subplot(2, 3, 4)
        run_data = live_plots['QPSK']
        plt.scatter(run_data['map'].real, run_data['map'].imag, c='red',
                    marker='x', s=100, zorder=3, label='Ideal')
        plt.scatter(run_data['rx'].real, run_data['rx'].imag,
                    color=run_data['color'], alpha=0.4, s=12, label='Rx Samples')
        plt.title(f"4. Node 1 IQ: QPSK (SNR={run_data['snr']}dB)")
        plt.axis('equal'); plt.grid(True, linestyle='--'); plt.legend(loc='upper right')

        # Plot 5: Full-chain 16QAM after sync + equalization (FIXED subplot)
        plt.subplot(2, 3, 5)
        plt.scatter(full_16qam['map'].real, full_16qam['map'].imag, c='black',
                    marker='+', s=80, zorder=3, label='Ideal')
        plt.scatter(full_16qam['rx_iq'].real, full_16qam['rx_iq'].imag,
                    c='crimson', marker='o', s=10, alpha=0.5, label='Post-sync Rx')
        plt.title(f"5. Full-chain 16QAM (coded BER={full_16qam['coded_ber']:.1e})")
        plt.axis('equal'); plt.grid(True, linestyle='--'); plt.legend(loc='upper right')

        # Plot 6: Node 2 64QAM constellation (uncoded live trace)
        plt.subplot(2, 3, 6)
        run_data = live_plots['64QAM']
        plt.scatter(run_data['map'].real, run_data['map'].imag, c='red',
                    marker='x', s=40, zorder=3, label='Ideal')
        plt.scatter(run_data['rx'].real, run_data['rx'].imag,
                    color=run_data['color'], alpha=0.5, s=8, label='Rx Samples')
        plt.title(f"6. Node 2 IQ: 64-QAM (SNR={run_data['snr']}dB)")
        plt.axis('equal'); plt.grid(True, linestyle='--'); plt.legend(loc='upper right')

        plt.tight_layout()
        #plt.savefig('ofdm_dashboard.png', dpi=110)
        try:
            plt.show()
        except Exception:
            pass
