/**
 * @file  fir_filter.c
 * @brief FIR filter using circular buffer — O(N) per sample
 */
#include "fir_filter.h"
#include <string.h>

void fir_init(FIRFilter *f, const float *coeffs, unsigned num_taps) {
    f->num_taps = (num_taps > FIR_MAX_TAPS) ? FIR_MAX_TAPS : num_taps;
    memcpy(f->coeffs, coeffs, f->num_taps * sizeof(float));
    fir_reset(f);
}

void fir_reset(FIRFilter *f) {
    memset(f->buf, 0, sizeof(f->buf));
    f->idx = 0;
}

float fir_update(FIRFilter *f, float sample) {
    f->buf[f->idx] = sample;
    f->idx = (f->idx + 1) % f->num_taps;
    float acc = 0.0f;
    unsigned i, j = f->idx;
    for (i = 0; i < f->num_taps; i++) {
        j = (j == 0) ? (f->num_taps - 1) : (j - 1);
        acc += f->coeffs[i] * f->buf[j];
    }
    return acc;
}

/*
 * Process a block of samples. Each input sample is shifted into the circular
 * buffer and one filtered output is produced per input sample. The output
 * buffer may alias the input buffer for in place processing.
 *
 *   in            pointer to input samples
 *   out           pointer to output buffer (length sample_length)
 *   sample_length number of samples to process
 */
void fir_update_block(FIRFilter *f, const float *in, float *out,
                      unsigned sample_length) {
    unsigned n;
    for (n = 0; n < sample_length; n++) {
        /* Shift the new sample into the circular buffer. */
        f->buf[f->idx] = in[n];
        f->idx = (f->idx + 1) % f->num_taps;

        /* Multiply accumulate across all taps, newest sample first. */
        float acc = 0.0f;
        unsigned i, j = f->idx;
        for (i = 0; i < f->num_taps; i++) {
            j = (j == 0) ? (f->num_taps - 1) : (j - 1);
            acc += f->coeffs[i] * f->buf[j];
        }
        out[n] = acc;
    }
}
