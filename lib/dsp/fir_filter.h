/**
 * @file  fir_filter.h
 * @brief FIR filter — C99, no stdlib, no dynamic allocation
 */
#ifndef FIR_FILTER_H
#define FIR_FILTER_H

#ifdef __cplusplus
extern "C" {
#endif

#define FIR_MAX_TAPS 128

typedef struct {
    float    coeffs[FIR_MAX_TAPS];
    float    buf[FIR_MAX_TAPS];
    unsigned num_taps;
    unsigned idx;
} FIRFilter;

void  fir_init(FIRFilter *f, const float *coeffs, unsigned num_taps);
void  fir_reset(FIRFilter *f);
float fir_update(FIRFilter *f, float sample);
void  fir_update_block(FIRFilter *f, const float *in, float *out,
                       unsigned sample_length);

#ifdef __cplusplus
}
#endif
#endif
