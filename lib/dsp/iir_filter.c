#include "iir_filter.h"

void iir_init(IIRFilter *f, float b0, float b1, float b2, float a1, float a2) {
    f->b0 = b0; f->b1 = b1; f->b2 = b2;
    f->a1 = a1; f->a2 = a2;
    iir_reset(f);
}

void iir_reset(IIRFilter *f) { f->w1 = f->w2 = 0.0f; }

float iir_update(IIRFilter *f, float x) {
    float y = f->b0 * x + f->w1;
    f->w1   = f->b1 * x - f->a1 * y + f->w2;
    f->w2   = f->b2 * x - f->a2 * y;
    return y;
}

/*
 * Process a block of samples through the biquad. One filtered output is
 * produced per input sample. The output buffer may alias the input buffer
 * for in place processing.
 *
 *   in            pointer to input samples
 *   out           pointer to output buffer (length sample_length)
 *   sample_length number of samples to process
 */
void iir_update_block(IIRFilter *f, const float *in, float *out,
                      unsigned sample_length) {
    unsigned n;
    for (n = 0; n < sample_length; n++) {
        float x = in[n];
        float y = f->b0 * x + f->w1;
        f->w1   = f->b1 * x - f->a1 * y + f->w2;
        f->w2   = f->b2 * x - f->a2 * y;
        out[n]  = y;
    }
}
