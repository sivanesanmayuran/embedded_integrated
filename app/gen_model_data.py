import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
from tensorflow import keras

# Minimal model: single Dense (FullyConnected) layer, 1 input -> 1 output.
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,), dtype=tf.float32),
    tf.keras.layers.Dense(1),
])

# Fix weights so inference is deterministic: y = 2*x + 1
model.layers[0].set_weights([np.array([[2.0]], dtype=np.float32),
                             np.array([1.0], dtype=np.float32)])

converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Keep as plain float32 TFLite (FULLY_CONNECTED builtin op).
tflite_model = converter.convert()

out_dir = r"C:\NXP\SW32N5_GRAYVIP_1.0.25.0_D2604\framework\common\components\tflm"

header = """#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#ifdef __cplusplus
extern "C" {
#endif

extern const unsigned char g_model[];
extern const unsigned int g_model_len;

#ifdef __cplusplus
}
#endif

#endif /* MODEL_DATA_H */
"""

with open(os.path.join(out_dir, "model_data.h"), "w") as f:
    f.write(header)

lines = []
lines.append('#include "model_data.h"')
lines.append("")
lines.append("const unsigned char g_model[] = {")
row = []
body = []
for i, b in enumerate(tflite_model):
    row.append("0x%02x" % b)
    if len(row) == 12:
        body.append("    " + ", ".join(row) + ",")
        row = []
if row:
    body.append("    " + ", ".join(row) + ",")
lines.extend(body)
lines.append("};")
lines.append("const unsigned int g_model_len = %d;" % len(tflite_model))
lines.append("")

with open(os.path.join(out_dir, "model_data.cc"), "w") as f:
    f.write("\n".join(lines))

print("Model size:", len(tflite_model), "bytes")
print("Wrote model_data.h and model_data.cc")
