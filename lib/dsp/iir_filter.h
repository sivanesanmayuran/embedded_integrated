/**
 * @file  iir_filter.h
 * @brief IIR biquad — Direct Form II Transposed, C99
 */
#ifndef IIR_FILTER_H
#define IIR_FILTER_H
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float b0, b1, b2;
    float a1, a2;
    float w1, w2;
} IIRFilter;

void  iir_init(IIRFilter *f, float b0, float b1, float b2, float a1, float a2);
void  iir_reset(IIRFilter *f);
float iir_update(IIRFilter *f, float sample);
void  iir_update_block(IIRFilter *f, const float *in, float *out,
                       unsigned sample_length);

#ifdef __cplusplus
}
#endif
#endif
