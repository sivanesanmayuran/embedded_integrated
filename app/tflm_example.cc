#include <stdio.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"

extern "C"
{
    int TFLM_Init(void);
    int TFLM_Run(float input, float *output);
}

namespace
{
    constexpr int kTensorArenaSize = 2048;

    uint8_t tensor_arena[kTensorArenaSize];

    const tflite::Model *model = nullptr;

    tflite::MicroInterpreter *interpreter = nullptr;

    TfLiteTensor *input_tensor = nullptr;
    TfLiteTensor *output_tensor = nullptr;
}

/*------------------------------------------------------------------*/
/* Init */
/*------------------------------------------------------------------*/
int TFLM_Init(void)
{
    model = tflite::GetModel(g_model);

    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        return -1;
    }

    static tflite::MicroMutableOpResolver<1> resolver;

    resolver.AddFullyConnected();

    static tflite::MicroInterpreter static_interpreter(
        model,
        resolver,
        tensor_arena,
        kTensorArenaSize);

    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        return -2;
    }

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    return 0;
}

/*------------------------------------------------------------------*/
/* Run inference */
/*------------------------------------------------------------------*/
int TFLM_Run(float input, float *output)
{
    if ((interpreter == nullptr) || (output == nullptr))
    {
        return -1;
    }

    input_tensor->data.f[0] = input;

    if (interpreter->Invoke() != kTfLiteOk)
    {
        return -2;
    }

    *output = output_tensor->data.f[0];

    return 0;
}