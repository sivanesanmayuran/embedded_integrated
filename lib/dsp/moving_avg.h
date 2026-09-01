/**
 * @file  moving_avg.h
 * @brief Simple and Exponential Moving Average — C99, no stdlib
 */
#ifndef MOVING_AVG_H
#define MOVING_AVG_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SMA_MAX_WINDOW 64

/* Simple Moving Average */
typedef struct {
    float    buffer[SMA_MAX_WINDOW];
    uint16_t window;
    uint16_t index;
    float    sum;
    uint16_t count;
} SMAFilter;

void  sma_init   (SMAFilter *f, uint16_t window);
void  sma_reset  (SMAFilter *f);
float sma_update (SMAFilter *f, float sample);

/* Exponential Moving Average */
typedef struct {
    float alpha;
    float value;
    uint8_t initialized;
} EMAFilter;

void  ema_init   (EMAFilter *f, float alpha);
void  ema_reset  (EMAFilter *f);
float ema_update (EMAFilter *f, float sample);

#ifdef __cplusplus
}
#endif
#endif /* MOVING_AVG_H */
