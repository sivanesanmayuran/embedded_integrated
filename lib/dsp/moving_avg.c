/**
 * @file  moving_avg.c
 */
#include "moving_avg.h"
#include <string.h>

/* ── SMA ─────────────────────────────────────────────────────────────── */

void sma_init(SMAFilter *f, uint16_t window)
{
    f->window = (window > SMA_MAX_WINDOW) ? SMA_MAX_WINDOW : window;
    sma_reset(f);
}

void sma_reset(SMAFilter *f)
{
    f->index = 0;
    f->sum   = 0.0f;
    f->count = 0;
    memset(f->buffer, 0, sizeof(f->buffer));
}

float sma_update(SMAFilter *f, float sample)
{
    f->sum -= f->buffer[f->index];
    f->buffer[f->index] = sample;
    f->sum += sample;
    f->index = (f->index + 1) % f->window;
    if (f->count < f->window) f->count++;
    return f->sum / f->count;
}

/* ── EMA ─────────────────────────────────────────────────────────────── */

void ema_init(EMAFilter *f, float alpha)
{
    f->alpha       = alpha;
    f->value       = 0.0f;
    f->initialized = 0;
}

void ema_reset(EMAFilter *f)
{
    f->value       = 0.0f;
    f->initialized = 0;
}

float ema_update(EMAFilter *f, float sample)
{
    if (!f->initialized) {
        f->value       = sample;
        f->initialized = 1;
    } else {
        f->value = f->alpha * sample + (1.0f - f->alpha) * f->value;
    }
    return f->value;
}
