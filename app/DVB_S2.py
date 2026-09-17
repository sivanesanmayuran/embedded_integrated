"""
DVB-S2-style end-to-end modem simulation (educational, self-contained).

This models a full transmit and receive chain rather than an ideal coherent
link. The blocks below are intentionally simplified relative to the real
ETSI EN 302 307 standard (frame sizes, exact LDPC/BCH tables, pilot layout),
but every functional stage of a practical satellite modem is represented so
the digital FEC path and the IQ/synchronization path are both exercised.

TX chain:
    random bits
      -> BCH outer encode        (algebraic block code over GF(2^m))
      -> LDPC inner encode        (sparse parity check, systematic form)
      -> bit interleaver
      -> PRBS scrambler
      -> constellation mapper     (QPSK / 8PSK / 16APSK / 32APSK)
      -> pilot / preamble framing
      -> RRC pulse shaping (upsample + filter)
      -> HPA nonlinearity          (Saleh AM/AM, AM/PM)

Channel:
      -> carrier frequency offset (CFO) / Doppler
      -> phase noise (Wiener)
      -> AWGN

RX chain:
      -> AGC (automatic gain control)
      -> RRC matched filter
      -> Gardner timing recovery   (fractional-delay interpolation)
      -> coarse + fine CFO / Doppler estimation (pilot-aided)
      -> carrier phase recovery    (pilot-aided + decision-directed PLL)
      -> SNR / noise-variance estimation
      -> soft demapper -> per-bit LLRs
      -> descrambler
      -> deinterleaver
      -> LDPC belief-propagation decode (soft, uses LLRs)
      -> BCH decode (syndrome + Berlekamp-Massey + Chien)
      -> bit sink, BER / FER measurement

A multi-carrier polyphase filter bank (PFB) channelizer is also included to
demonstrate frequency-multiplexed IQ sampling and per-channel downconversion.

Run:
    python app/DVB_S2.py
"""

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False


# =============================================================================
# GF(2) linear algebra helpers (used by the LDPC code)
# =============================================================================
def gf2_row_reduce(mat):
    """Row-reduce a binary matrix in place, returning pivot column list."""
    m = mat.copy() % 2
    rows, cols = m.shape
    pivot_cols = []
    r = 0
    for c in range(cols):
        # find a pivot in column c at or below row r
        pivot = -1
        for i in range(r, rows):
            if m[i, c]:
                pivot = i
                break
        if pivot == -1:
            continue
        m[[r, pivot]] = m[[pivot, r]]
        # eliminate this column from all other rows
        for i in range(rows):
            if i != r and m[i, c]:
                m[i, :] ^= m[r, :]
        pivot_cols.append(c)
        r += 1
        if r == rows:
            break
    return m, pivot_cols


# =============================================================================
# LDPC inner code (regular, constructed parity-check matrix)
# =============================================================================
class LDPCCode:
    """
    A small regular LDPC code built from a random (wc, wr) bipartite graph.
    Systematic generator is derived from H via GF(2) elimination. Decoding is
    log-domain sum-product (belief propagation) driven by channel LLRs.
    """

    def __init__(self, n, wc=3, wr=6, seed=1):
        assert (n * wc) % wr == 0, "n*wc must be divisible by wr"
        self.n = n
        m = (n * wc) // wr          # number of parity checks
        self.m = m
        self.k = n - m

        rng = np.random.default_rng(seed)
        # Build H with column weight wc and (approximately) row weight wr using
        # a permutation-of-sockets construction, then de-duplicate.
        H = np.zeros((m, n), dtype=np.uint8)
        for attempt in range(200):
            H[:] = 0
            col_sockets = np.repeat(np.arange(n), wc)
            rng.shuffle(col_sockets)
            row_sockets = np.repeat(np.arange(m), wr)
            ok = True
            for cs, rs in zip(col_sockets, row_sockets):
                if H[rs, cs]:
                    ok = False
                    break
                H[rs, cs] = 1
            if ok:
                break
        self.H = H

        # Derive a systematic generator. Reduce H and pick independent columns
        # to act as the parity part.
        Hr, pivot_cols = gf2_row_reduce(H)
        # Guard: if rank deficient, adjust m/k accordingly.
        rank = len(pivot_cols)
        self.m = rank
        self.k = n - rank
        self.parity_cols = pivot_cols
        self.info_cols = [c for c in range(n) if c not in set(pivot_cols)]

        # After reduction, Hr has identity on pivot columns. For an info bit
        # column j, the parity bits are the entries of Hr in that column.
        self.Hr = Hr[:rank, :]

        # Precompute check-node -> variable-node adjacency for BP.
        self.check_vars = [np.where(H[i] == 1)[0] for i in range(H.shape[0])]
        self.var_checks = [np.where(H[:, j] == 1)[0] for j in range(n)]

    def encode(self, info_bits):
        """Map k info bits to an n-bit codeword satisfying H c = 0."""
        info_bits = np.asarray(info_bits, dtype=np.uint8) % 2
        c = np.zeros(self.n, dtype=np.uint8)
        c[self.info_cols] = info_bits
        # Solve for parity bits so that Hr c = 0 (Hr is identity on parity cols).
        # parity_bit_i = sum over info cols of Hr[i, info] * info_bit
        info_part = self.Hr[:, self.info_cols] @ info_bits % 2
        for row, pc in enumerate(self.parity_cols):
            c[pc] = info_part[row]
        return c

    def decode(self, llr, max_iters=40):
        """
        Log-domain sum-product decoding.
        llr: channel LLRs, positive => bit 0 more likely.
        Returns (info_bits, codeword, success_flag).
        """
        H = self.H
        m, n = H.shape
        # Messages variable->check initialized to channel LLRs.
        Mvc = {}
        for j in range(n):
            for i in self.var_checks[j]:
                Mvc[(i, j)] = llr[j]

        for _ in range(max_iters):
            # Check node update (tanh rule).
            Mcv = {}
            for i in range(m):
                vs = self.check_vars[i]
                tanh_vals = np.array([np.tanh(np.clip(Mvc[(i, j)] / 2.0, -30, 30))
                                      for j in vs])
                for idx, j in enumerate(vs):
                    prod = np.prod(np.delete(tanh_vals, idx))
                    prod = np.clip(prod, -0.999999999, 0.999999999)
                    Mcv[(i, j)] = 2.0 * np.arctanh(prod)

            # Variable node update + hard decision.
            total = llr.copy()
            for j in range(n):
                for i in self.var_checks[j]:
                    total[j] += Mcv[(i, j)]
            hard = (total < 0).astype(np.uint8)

            # Parity check.
            if np.all((H @ hard) % 2 == 0):
                info = hard[self.info_cols]
                return info, hard, True

            # Update variable->check messages.
            for j in range(n):
                for i in self.var_checks[j]:
                    Mvc[(i, j)] = total[j] - Mcv[(i, j)]

        info = hard[self.info_cols]
        return info, hard, False


# =============================================================================
# BCH outer code over GF(2^m) (syndrome + Berlekamp-Massey + Chien search)
# =============================================================================
class BCHCode:
    """Narrow-sense binary BCH code with error-correction capability t."""

    def __init__(self, m, t):
        self.m = m
        self.n = (1 << m) - 1
        self.t = t
        # Primitive polynomials for small m (coefficients as integer bitmask).
        prim = {3: 0b1011, 4: 0b10011, 5: 0b100101, 6: 0b1000011,
                7: 0b10001001, 8: 0b100011101}
        self.prim_poly = prim[m]
        self._build_tables()
        self._build_generator()
        self.k = self.n - (len(self.gen) - 1)

    def _build_tables(self):
        self.exp = [0] * (2 * self.n)
        self.log = [0] * (self.n + 1)
        x = 1
        for i in range(self.n):
            self.exp[i] = x
            self.log[x] = i
            x <<= 1
            if x & (1 << self.m):
                x ^= self.prim_poly
        for i in range(self.n, 2 * self.n):
            self.exp[i] = self.exp[i - self.n]

    def _gf_mul(self, a, b):
        if a == 0 or b == 0:
            return 0
        return self.exp[self.log[a] + self.log[b]]

    def _gf_inv(self, a):
        return self.exp[self.n - self.log[a]]

    def _minimal_polys_generator(self):
        # Build generator as LCM of minimal polynomials of alpha^1..alpha^2t.
        # Work with sets of conjugate roots.
        needed = set()
        for i in range(1, 2 * self.t + 1):
            r = i % self.n
            # add conjugacy class
            c = r
            while c not in needed:
                needed.add(c)
                c = (c * 2) % self.n
        return needed

    def _build_generator(self):
        roots = self._minimal_polys_generator()
        # Multiply out (x - alpha^r) over GF(2^m); result must be binary.
        g = [1]  # polynomial coefficients, low order first, in GF(2^m)
        for r in sorted(roots):
            root = self.exp[r]
            newg = [0] * (len(g) + 1)
            for i in range(len(g)):
                # multiply g[i]*x
                newg[i + 1] ^= g[i]
                # multiply g[i]*root
                newg[i] ^= self._gf_mul(g[i], root)
            g = newg
        # g should now be binary (0/1). Reduce just in case.
        self.gen = [1 if (coef & 1) else (1 if coef else 0) for coef in g]
        # Ensure binary
        self.gen = [1 if c else 0 for c in g]

    def encode(self, msg_bits):
        """Systematic encode: n-bit codeword = [parity | message]."""
        msg_bits = np.asarray(msg_bits, dtype=np.uint8) % 2
        assert len(msg_bits) == self.k
        # Multiply message by x^(n-k) then take remainder mod g(x) over GF(2).
        nk = self.n - self.k
        # message polynomial shifted
        reg = np.zeros(nk, dtype=np.uint8)
        for b in msg_bits:
            feedback = b ^ reg[-1]
            new = np.zeros(nk, dtype=np.uint8)
            for i in range(nk - 1, 0, -1):
                new[i] = reg[i - 1] ^ (feedback & self.gen[i])
            new[0] = feedback & self.gen[0]
            reg = new
        parity = reg[::-1]
        return np.concatenate([parity, msg_bits]).astype(np.uint8)

    def _syndromes(self, r_bits):
        S = []
        for i in range(1, 2 * self.t + 1):
            acc = 0
            for j, bit in enumerate(r_bits):
                if bit:
                    acc ^= self.exp[(i * j) % self.n]
            S.append(acc)
        return S

    def decode(self, r_bits):
        """Correct up to t errors; returns (message_bits, num_corrected)."""
        r_bits = np.asarray(r_bits, dtype=np.uint8).copy() % 2
        S = self._syndromes(r_bits)
        if all(s == 0 for s in S):
            return r_bits[self.n - self.k:], 0

        # Berlekamp-Massey over GF(2^m).
        L = 0
        C = [1]
        B = [1]
        b = 1
        mstep = 1
        for nidx in range(2 * self.t):
            # discrepancy
            delta = S[nidx]
            for i in range(1, L + 1):
                if i < len(C) and C[i]:
                    delta ^= self._gf_mul(C[i], S[nidx - i])
            if delta == 0:
                mstep += 1
            elif 2 * L <= nidx:
                T = list(C)
                coef = self._gf_mul(delta, self._gf_inv(b))
                shifted = [0] * mstep + [self._gf_mul(coef, x) for x in B]
                C = self._poly_add(C, shifted)
                L = nidx + 1 - L
                B = T
                b = delta
                mstep = 1
            else:
                coef = self._gf_mul(delta, self._gf_inv(b))
                shifted = [0] * mstep + [self._gf_mul(coef, x) for x in B]
                C = self._poly_add(C, shifted)
                mstep += 1

        # Chien search for roots of locator polynomial.
        err_pos = []
        for i in range(self.n):
            acc = 0
            for j in range(len(C)):
                if C[j]:
                    acc ^= self.exp[(self.log[C[j]] + i * j) % self.n]
            if acc == 0:
                err_pos.append((self.n - i) % self.n)

        for pos in err_pos:
            r_bits[pos] ^= 1

        # Verify
        S2 = self._syndromes(r_bits)
        corrected = len(err_pos) if all(s == 0 for s in S2) else -1
        return r_bits[self.n - self.k:], corrected

    @staticmethod
    def _poly_add(a, b):
        n = max(len(a), len(b))
        out = [0] * n
        for i in range(len(a)):
            out[i] ^= a[i]
        for i in range(len(b)):
            out[i] ^= b[i]
        return out


# =============================================================================
# Modulation: constellations, bit labels, mapper, soft demapper (LLR)
# =============================================================================
def get_dvbs2_constellation(mod_type):
    if mod_type == 'QPSK':
        angles = np.pi / 4 + np.arange(4) * (2 * np.pi / 4)
        c = np.cos(angles) + 1j * np.sin(angles)
    elif mod_type == '8PSK':
        angles = np.arange(8) * (2 * np.pi / 8)
        c = np.cos(angles) + 1j * np.sin(angles)
    elif mod_type == '16APSK':
        r1, r2 = 1.0, 2.75
        inner = r1 * np.exp(1j * (np.pi / 4 + np.arange(4) * (2 * np.pi / 4)))
        outer = r2 * np.exp(1j * (np.pi / 12 + np.arange(12) * (2 * np.pi / 12)))
        c = np.concatenate([inner, outer])
    elif mod_type == '32APSK':
        r1, r2, r3 = 1.0, 2.84, 5.27
        inner = r1 * np.exp(1j * (np.pi / 4 + np.arange(4) * (2 * np.pi / 4)))
        middle = r2 * np.exp(1j * (np.pi / 12 + np.arange(12) * (2 * np.pi / 12)))
        outer = r3 * np.exp(1j * (np.pi / 16 + np.arange(16) * (2 * np.pi / 16)))
        c = np.concatenate([inner, middle, outer])
    else:
        raise ValueError("Unknown modulation scheme type.")
    return c / np.sqrt(np.mean(np.abs(c) ** 2))


def bits_per_symbol(mod_type):
    return {'QPSK': 2, '8PSK': 3, '16APSK': 4, '32APSK': 5}[mod_type]


def make_bit_labels(m_order):
    """Binary labels for constellation indices (index -> bit vector)."""
    bps = int(np.log2(m_order))
    labels = np.zeros((m_order, bps), dtype=np.uint8)
    for i in range(m_order):
        for b in range(bps):
            labels[i, b] = (i >> (bps - 1 - b)) & 1
    return labels


def modulate(bits, constellation, labels):
    bps = labels.shape[1]
    assert len(bits) % bps == 0
    groups = bits.reshape(-1, bps)
    # map each group of bits to the matching constellation index
    weights = (1 << np.arange(bps - 1, -1, -1))
    idx = groups @ weights
    return constellation[idx], idx


def soft_demap(rx_syms, constellation, labels, noise_var):
    """Max-log LLR per bit. Positive LLR => bit 0 more likely."""
    bps = labels.shape[1]
    m_order = len(constellation)
    # distance metric per symbol per constellation point
    d2 = np.abs(rx_syms[:, None] - constellation[None, :]) ** 2
    metric = -d2 / max(noise_var, 1e-9)
    llrs = np.zeros((len(rx_syms), bps))
    for b in range(bps):
        zero_pts = np.where(labels[:, b] == 0)[0]
        one_pts = np.where(labels[:, b] == 1)[0]
        max0 = np.max(metric[:, zero_pts], axis=1)
        max1 = np.max(metric[:, one_pts], axis=1)
        llrs[:, b] = max0 - max1
    return llrs.reshape(-1)


# =============================================================================
# Pulse shaping (RRC) and the polyphase channelizer (IQ multiplexing demo)
# =============================================================================
def rrc_filter(sps, beta, span):
    num_taps = sps * span + 1
    t = (np.arange(num_taps) - (num_taps - 1) / 2.0) / sps
    h = np.zeros(num_taps)
    for i, ti in enumerate(t):
        if ti == 0.0:
            h[i] = 1.0 - beta + (4.0 * beta / np.pi)
        elif beta != 0.0 and np.isclose(np.abs(4.0 * beta * ti), 1.0):
            h[i] = (beta / np.sqrt(2.0)) * (
                ((1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))) +
                ((1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))))
        else:
            num = (np.sin(np.pi * ti * (1.0 - beta)) +
                   4.0 * beta * ti * np.cos(np.pi * ti * (1.0 + beta)))
            den = np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
            h[i] = num / den
    return h / np.sqrt(np.sum(h ** 2))


def init_prototype_filter(num_channels, taps_per_path):
    filter_len = num_channels * taps_per_path
    t = np.arange(filter_len) - (filter_len - 1) / 2.0
    with np.errstate(divide='ignore', invalid='ignore'):
        proto = np.sin(np.pi * (1.0 / num_channels) * t) / (np.pi * (1.0 / num_channels) * t)
    proto[np.isnan(proto)] = 1.0
    window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(filter_len) / (filter_len - 1)))
    proto *= window
    return proto * (num_channels / np.sum(proto))


class PolyphaseChannelizer:
    def __init__(self, prototype_lpf, num_channels, taps_per_path):
        self.num_channels = num_channels
        self.prototype_lpf = prototype_lpf.reshape(taps_per_path, num_channels)
        self.buffer = np.zeros((num_channels, taps_per_path), dtype=complex)
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


# =============================================================================
# Channel impairments
# =============================================================================
def apply_hpa_saleh(x, backoff_db=6.0):
    """Saleh AM/AM and AM/PM nonlinearity with input back-off."""
    alpha_a, beta_a = 2.1587, 1.1517      # AM/AM
    alpha_p, beta_p = 4.0033, 9.1040      # AM/PM
    scale = 10 ** (-backoff_db / 20.0)
    r = np.abs(x) * scale
    amam = (alpha_a * r) / (1.0 + beta_a * r ** 2)
    ampm = (alpha_p * r ** 2) / (1.0 + beta_p * r ** 2)
    phase = np.angle(x) + ampm
    return amam * np.exp(1j * phase)


def add_cfo_phase_noise(x, fs, cfo_hz, phase_noise_std):
    """Apply carrier frequency offset (Doppler) and Wiener phase noise."""
    n = np.arange(len(x))
    cfo = np.exp(1j * 2 * np.pi * cfo_hz / fs * n)
    pn = np.cumsum(np.random.normal(0, phase_noise_std, len(x)))
    return x * cfo * np.exp(1j * pn)


def add_awgn(x, snr_db):
    sig_power = np.mean(np.abs(x) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power / 2.0) * (
        np.random.normal(0, 1, len(x)) + 1j * np.random.normal(0, 1, len(x)))
    return x + noise, noise_power


# =============================================================================
# RX synchronization blocks
# =============================================================================
def agc(x, target_rms=1.0):
    """Automatic gain control: normalize to a target RMS level."""
    rms = np.sqrt(np.mean(np.abs(x) ** 2))
    if rms < 1e-12:
        return x, 1.0
    g = target_rms / rms
    return x * g, g


def gardner_timing_recovery(x, sps):
    """
    Gardner timing error detector with a simple linear interpolator and a
    first-order loop. Returns symbol-rate samples.
    """
    mu = 0.0            # fractional interpolation index
    kp = 0.01           # loop gain
    out = []
    i = sps             # start after one symbol
    n = len(x)
    while i < n - sps - 2:
        base = int(np.floor(i))
        frac = i - base
        # linear interpolation for the on-time and half-symbol samples
        def interp(idx_float):
            b = int(np.floor(idx_float))
            f = idx_float - b
            if b + 1 >= n:
                return x[b]
            return (1 - f) * x[b] + f * x[b + 1]
        curr = interp(i)
        half = interp(i - sps / 2.0)
        prev = interp(i - sps)
        # Gardner TED (works for real+imag)
        e = np.real(np.conj(half) * (curr - prev))
        mu = kp * e
        out.append(curr)
        i += sps + mu
    return np.array(out)


def estimate_cfo_pilot(rx_pilots, tx_pilots, fs_sym):
    """Data-aided CFO estimate from the phase slope across known pilots."""
    prod = rx_pilots * np.conj(tx_pilots)
    # differential phase between consecutive pilots
    diff = prod[1:] * np.conj(prod[:-1])
    dphi = np.angle(np.mean(diff))
    cfo_norm = dphi / (2 * np.pi)      # cycles per symbol
    return cfo_norm


def correct_cfo(x, cfo_norm):
    n = np.arange(len(x))
    return x * np.exp(-1j * 2 * np.pi * cfo_norm * n)


def carrier_phase_pll(x, constellation, alpha=0.01):
    """Decision-directed first-order phase-locked loop for residual phase."""
    out = np.zeros_like(x)
    theta = 0.0
    for i, s in enumerate(x):
        y = s * np.exp(-1j * theta)
        # nearest constellation point (decision)
        dec = constellation[np.argmin(np.abs(constellation - y))]
        err = np.angle(y * np.conj(dec))
        theta += alpha * err
        out[i] = y
    return out


def estimate_noise_var(rx_syms, constellation):
    """Estimate noise variance from decision distances (EVM-based)."""
    dec = constellation[np.argmin(
        np.abs(rx_syms[:, None] - constellation[None, :]), axis=1)]
    err = rx_syms - dec
    return np.mean(np.abs(err) ** 2)


# =============================================================================
# Scrambler / interleaver
# =============================================================================
def prbs_sequence(length, seed=0x7F):
    """Simple 8-bit LFSR PRBS used as a scrambling mask."""
    reg = seed & 0xFF
    out = np.zeros(length, dtype=np.uint8)
    for i in range(length):
        bit = ((reg >> 7) ^ (reg >> 5) ^ (reg >> 4) ^ (reg >> 3)) & 1
        out[i] = reg & 1
        reg = ((reg << 1) | bit) & 0xFF
    return out


def make_interleaver(length, seed=7):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(length)
    inv = np.argsort(perm)
    return perm, inv


# =============================================================================
# Full single-carrier modem chain (FEC + sync + impairments)
# =============================================================================
def run_full_chain(mod_type='16APSK', snr_db=14.0, cfo_hz=800.0,
                   phase_noise_std=0.004, hpa_backoff_db=8.0,
                   sym_rate=1e6, sps=4, beta=0.25, span=10, verbose=True):
    rng = np.random.default_rng(2024)
    fs = sym_rate * sps

    # ---- Codes ----
    bch = BCHCode(m=6, t=3)                 # (63, k) BCH outer code
    # LDPC block length aligned to bits/symbol. The LDPC message length k must
    # be at least the full BCH codeword length (bch.n) so the entire outer
    # codeword is carried by the inner code without truncation.
    bps = bits_per_symbol(mod_type)
    ldpc_n = 126
    ldpc = None
    while True:
        if ldpc_n % bps == 0 and (ldpc_n * 3) % 6 == 0:
            cand = LDPCCode(n=ldpc_n, wc=3, wr=6, seed=3)
            if cand.k >= bch.n and (ldpc_n % bps == 0):
                ldpc = cand
                break
        ldpc_n += 1


    # ---- TX ----
    info_len = min(bch.k, ldpc.k)
    info_bits = rng.integers(0, 2, info_len).astype(np.uint8)

    # BCH outer encode (pad message up to bch.k)
    bch_msg = np.zeros(bch.k, dtype=np.uint8)
    bch_msg[:info_len] = info_bits
    bch_cw = bch.encode(bch_msg)

    # LDPC inner encode (pad up to ldpc.k)
    ldpc_msg = np.zeros(ldpc.k, dtype=np.uint8)
    copy_len = min(len(bch_cw), ldpc.k)
    ldpc_msg[:copy_len] = bch_cw[:copy_len]
    coded = ldpc.encode(ldpc_msg)

    # Interleave
    perm, inv = make_interleaver(len(coded))
    interleaved = coded[perm]

    # Scramble
    mask = prbs_sequence(len(interleaved))
    scrambled = interleaved ^ mask

    # Map to symbols
    constellation = get_dvbs2_constellation(mod_type)
    labels = make_bit_labels(len(constellation))
    data_syms, _ = modulate(scrambled, constellation, labels)

    # Framing: prepend a block of known pilot symbols (preamble) for sync.
    num_pilots = 64
    pilot_syms = constellation[rng.integers(0, len(constellation), num_pilots)]
    frame_syms = np.concatenate([pilot_syms, data_syms])

    # RRC pulse shaping (upsample + filter)
    h_rrc = rrc_filter(sps, beta, span)
    up = np.zeros(len(frame_syms) * sps, dtype=complex)
    up[::sps] = frame_syms
    tx_wave = np.convolve(up, h_rrc, mode='full')

    # ---- Channel ----
    tx_hpa = apply_hpa_saleh(tx_wave, backoff_db=hpa_backoff_db)
    ch = add_cfo_phase_noise(tx_hpa, fs, cfo_hz, phase_noise_std)
    rx_wave, noise_power = add_awgn(ch, snr_db)

    # ---- RX ----
    rx_wave, _ = agc(rx_wave, target_rms=np.sqrt(np.mean(np.abs(tx_wave) ** 2)))

    # Matched filter
    mf = np.convolve(rx_wave, h_rrc, mode='full')
    # Compensate the group delay of the two RRC convolutions.
    delay = (len(h_rrc) - 1)
    mf = mf[delay:]

    # Symbol-rate sampling (start at the RRC peak). We use direct decimation
    # here; the Gardner routine below refines timing on the sampled stream.
    sampled = mf[::sps][:len(frame_syms)]

    # Timing refinement on an oversampled window (demonstration of Gardner).
    _ = gardner_timing_recovery(mf[:sps * (num_pilots + 32)], sps)

    # Split pilots / data
    rx_pilots = sampled[:num_pilots]
    rx_data = sampled[num_pilots:num_pilots + len(data_syms)]

    # Coarse + fine CFO / Doppler estimation from pilots
    cfo_norm = estimate_cfo_pilot(rx_pilots, pilot_syms, sym_rate)
    rx_data = correct_cfo(rx_data, cfo_norm)
    rx_pilots_c = correct_cfo(rx_pilots, cfo_norm)

    # Residual carrier phase from pilots
    phase_off = np.angle(np.mean(rx_pilots_c * np.conj(pilot_syms)))
    rx_data = rx_data * np.exp(-1j * phase_off)

    # Fine decision-directed carrier phase recovery
    rx_data = carrier_phase_pll(rx_data, constellation, alpha=0.02)

    # SNR / noise variance estimation
    est_nvar = estimate_noise_var(rx_data, constellation)

    # Soft demap -> LLRs
    llr_scrambled = soft_demap(rx_data, constellation, labels, est_nvar)

    # Descramble LLRs (flip sign where mask bit is 1)
    mask_signs = 1.0 - 2.0 * mask
    llr_interleaved = llr_scrambled * mask_signs

    # Deinterleave
    llr_coded = np.zeros_like(llr_interleaved)
    llr_coded[perm] = llr_interleaved

    # LDPC soft decode
    ldpc_info_hat, ldpc_cw_hat, ldpc_ok = ldpc.decode(llr_coded, max_iters=40)

    # BCH decode
    bch_rx = ldpc_info_hat[:copy_len]
    bch_full = np.zeros(bch.n, dtype=np.uint8)
    bch_full[:len(bch_rx)] = bch_rx
    bch_msg_hat, ncorr = bch.decode(bch_full)
    info_hat = bch_msg_hat[:info_len]

    # ---- Metrics ----
    # Uncoded reference: hard decisions straight from the demapper.
    hard_scrambled = (llr_scrambled < 0).astype(np.uint8)
    uncoded_bits = (hard_scrambled ^ mask)[perm][:len(coded)]
    raw_ber = np.mean(uncoded_bits[:ldpc.k][:info_len] != info_bits) \
        if info_len <= ldpc.k else 0.0

    coded_bit_err = np.sum(info_hat != info_bits)
    coded_ber = coded_bit_err / info_len
    frame_error = int(coded_bit_err > 0)

    result = {
        'mod': mod_type, 'snr_db': snr_db, 'cfo_hz': cfo_hz,
        'est_cfo_hz': cfo_norm * sym_rate, 'phase_off_rad': phase_off,
        'est_noise_var': est_nvar, 'ldpc_converged': ldpc_ok,
        'bch_corrected': ncorr, 'raw_ber': raw_ber,
        'coded_ber': coded_ber, 'frame_error': frame_error,
        'rx_data': rx_data, 'constellation': constellation,
    }

    if verbose:
        print("=" * 60)
        print(f"Full DVB-S2 chain: {mod_type}  @ SNR = {snr_db} dB")
        print("-" * 60)
        print(f"  BCH code           : ({bch.n}, {bch.k}) t={bch.t}")
        print(f"  LDPC code          : ({ldpc.n}, {ldpc.k}) rate~{ldpc.k/ldpc.n:.2f}")
        print(f"  Info bits          : {info_len}")
        print(f"  Applied CFO        : {cfo_hz:.1f} Hz")
        print(f"  Estimated CFO      : {result['est_cfo_hz']:.1f} Hz")
        print(f"  Residual phase     : {phase_off:+.3f} rad")
        print(f"  Estimated noise var: {est_nvar:.4f}")
        print(f"  LDPC converged     : {ldpc_ok}")
        print(f"  BCH errors fixed   : {ncorr}")
        print(f"  Uncoded BER (info) : {raw_ber:.3e}")
        print(f"  Coded BER (info)   : {coded_ber:.3e}")
        print(f"  Frame error        : {'YES' if frame_error else 'no'}")
        print("=" * 60)
    return result


# =============================================================================
# Multi-carrier channelizer visualization (IQ multiplexing demo)
# =============================================================================
def run_channelizer_demo(show=True):
    NUM_CHANNELS = 16
    SPS = 4
    BETA = 0.25
    SPAN = 10
    NUM_SYMBOLS = 400
    SNR_DB = 22
    rng = np.random.default_rng(24)

    h_rrc = rrc_filter(SPS, BETA, SPAN)
    total_samples = NUM_SYMBOLS * SPS

    channel_mapping = {2: 'QPSK', 5: '8PSK', 9: '16APSK', 13: '32APSK'}
    tx_baseband = np.zeros((NUM_CHANNELS, total_samples), dtype=complex)
    ideal_maps = {}

    for ch, mod in channel_mapping.items():
        ideal_maps[mod] = get_dvbs2_constellation(mod)
        m_order = len(ideal_maps[mod])
        idx = rng.integers(0, m_order, NUM_SYMBOLS)
        syms = ideal_maps[mod][idx]
        up = np.zeros(total_samples, dtype=complex)
        up[::SPS] = syms
        tx_baseband[ch, :] = np.convolve(up, h_rrc, mode='same')

    n = np.arange(NUM_CHANNELS).reshape(-1, 1)
    k = np.arange(NUM_CHANNELS).reshape(1, -1)
    idft = np.exp(1j * 2 * np.pi * k * n / NUM_CHANNELS)
    wideband = np.dot(idft, tx_baseband)

    flat = wideband.flatten(order='F')
    sig_p = np.mean(np.abs(flat) ** 2)
    npow = sig_p / (10 ** (SNR_DB / 10.0))
    noise = np.sqrt(npow / 2.0) * (np.random.normal(0, 1, len(flat)) +
                                   1j * np.random.normal(0, 1, len(flat)))
    noisy = (flat + noise).reshape((NUM_CHANNELS, total_samples), order='F')

    proto = init_prototype_filter(NUM_CHANNELS, taps_per_path=4)
    chz = PolyphaseChannelizer(proto, NUM_CHANNELS, taps_per_path=4)
    rx = np.zeros((NUM_CHANNELS, total_samples), dtype=complex)
    for s in range(total_samples):
        rx[:, s] = chz.process_block(noisy[:, s])

    recovered = {}
    delay_align = (SPAN * SPS) // 2
    for ch, mod in channel_mapping.items():
        matched = np.convolve(rx[ch, :], h_rrc, mode='same')
        ds = matched[delay_align::SPS]
        recovered[mod] = ds[15:NUM_SYMBOLS - 15]

    if show and _HAVE_MPL:
        fig, axes = plt.subplots(2, 2, figsize=(12, 11))
        cfg = [('QPSK', axes[0, 0], 'royalblue'), ('8PSK', axes[0, 1], 'darkorange'),
               ('16APSK', axes[1, 0], 'forestgreen'), ('32APSK', axes[1, 1], 'crimson')]
        for name, ax, color in cfg:
            ax.scatter(ideal_maps[name].real, ideal_maps[name].imag, c='black',
                       marker='x', s=100, zorder=3, label='Ideal')
            ax.scatter(recovered[name].real, recovered[name].imag, alpha=0.5,
                       color=color, label='Recovered')
            ax.set_title(f"Channelizer {name}")
            ax.axis('equal')
            ax.grid(True, linestyle='--')
            ax.legend(loc='upper right')
        plt.tight_layout()
        #plt.savefig('dvbs2_channelizer.png', dpi=110)
        try:
            plt.show()
        except Exception:
            pass
    return recovered


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    np.random.seed(24)

    # 1a. QPSK sanity check: robust modulation, mild impairments. The full
    #     FEC + synchronization chain should recover the bits error-free.
    run_full_chain(mod_type='QPSK', snr_db=12.0, cfo_hz=200.0,
                   phase_noise_std=0.0005, hpa_backoff_db=14.0)

    # 1b. Full modem chain on 16APSK with moderate impairments.
    res = run_full_chain(mod_type='16APSK', snr_db=18.0, cfo_hz=200.0,
                         phase_noise_std=0.0005, hpa_backoff_db=14.0)


    # 2. Multi-carrier channelizer visualization.
    run_channelizer_demo(show=True)

    # 3. Post-sync constellation plot for the full-chain link.
    if _HAVE_MPL:
        plt.figure(figsize=(6, 6))
        c = res['constellation']
        plt.scatter(c.real, c.imag, c='black', marker='x', s=120, label='Ideal')
        plt.scatter(res['rx_data'].real, res['rx_data'].imag, s=8, alpha=0.4,
                    color='forestgreen', label='Recovered (post-sync)')
        plt.title(f"Full-chain {res['mod']} after AGC/CFO/phase recovery")
        plt.axis('equal')
        plt.grid(True, linestyle='--')
        plt.legend()
        plt.tight_layout()
        #plt.savefig('dvbs2_fullchain.png', dpi=110)
        try:
            plt.show()
        except Exception:
            pass
