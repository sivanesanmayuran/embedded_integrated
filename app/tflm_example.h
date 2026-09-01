#ifndef TFLM_HELLO_WORLD_H
#define TFLM_HELLO_WORLD_H

#ifdef __cplusplus
extern "C" {
#endif

int TFLM_Init(void);

int TFLM_Run(float input,
             float *output);

#ifdef __cplusplus
}
#endif

#endif
