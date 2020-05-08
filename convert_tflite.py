
# https://www.tensorflow.org/lite/guide/get_started#2_convert_the_model_format
if False:
    # https://code.visualstudio.com/docs/python/debugging#_remote-debugging
    # Launch applicaiton on remote computer: 
    # > python3 -m ptvsd --host 10.150.41.30 --port 3000 --wait convert_tflite.py
    import ptvsd
    # Allow other computers to attach to ptvsd at this IP address and port.
    ptvsd.enable_attach(address=('0.0.0.0', 3000), redirect_output=True)
    # Pause the program until a remote debugger is attached
    print("Wait for debugger attach")
    ptvsd.wait_for_attach()

import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model/1588860197')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
open("./tflite/1588860197.tflite", "wb").write(tflite_model)