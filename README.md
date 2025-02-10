# Moonshine TFLite

TensorFlow Lite port of Moonshine speech to text models

## Files

TensorFlow Lite versions of the Moonshine models are [available on HuggingFace](https://huggingface.co/UsefulSensors/moonshine/tree/main/tflite). They will also be automatically downloaded if you run the `transcribe.py` script.

## Running Moonshine

The `transcribe.py` script gives an example of how to run speech recognition in Python using the TFLite interpreter. It uses the class definition in `model.py`, and takes three optional arguments:

 - The path to a WAV file containing audio that you want to convert into text. This defaults to `assets/beckett.wav` if not specified.

 - The name of the model to use, either `moonshine/tiny` or `moonshine/base`. Defaults. to `moonshine/base`.

 - Path to a folder containing the four model files necessary for inference. If none is specified, defaults to downloading the files from HuggingFace.

 ## Converting from Keras

 The `convert.py` script runs an export process to convert the models from Keras format to TFLite. Currently it only supports float32 models. You shouldn't need to run this yourself unless you've modified the original Keras model, since the generated files are [available on HuggingFace](https://huggingface.co/UsefulSensors/moonshine/tree/main/tflite).