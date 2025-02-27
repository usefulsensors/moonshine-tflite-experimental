import numpy as np
import re
#import tensorflow as tf
import ai_edge_litert.interpreter as tflite


def _get_tflite_weights(model_name, precision="float"):
    from huggingface_hub import hf_hub_download

    if model_name not in ["tiny", "base"]:
        raise ValueError(f'Unknown model "{model_name}"')
    repo = "UsefulSensors/moonshine"
    subfolder = f"tflite/{model_name}/{precision}"

    return (
        hf_hub_download(repo, f"{x}.tfl", subfolder=subfolder)
        for x in ("decoder_initial", "decoder", "encoder", "preprocessor")
    )


class MoonshineTFLiteModel(object):
    def __init__(self, model_dir=None, model_name=None, model_precision="float"): 
        assert model_name is not None, (
            "model_name should always be specified"
        )
        assert model_name in ["moonshine/tiny", "moonshine/base"], (
            "model_name should be 'moonshine/tiny' or 'moonshine/base'"
        )
        self.model_name = model_name.split("/")[-1]
    
        assert self.model_name in ["tiny", "base"]
        if self.model_name == "tiny":
            self.layer_count = 6
        elif self.model_name == "base":
            self.layer_count = 8

        self.decoder_uncached_input_names = [
            "tokens",
            "audio_features",
            "seq_len",
        ]

        self.decoder_cached_input_names = self.decoder_uncached_input_names.copy()
        for index in range(self.layer_count):
            self.decoder_cached_input_names.append(f"input_cache_k_{index}")
            self.decoder_cached_input_names.append(f"input_cache_v_{index}")
            self.decoder_cached_input_names.append(f"input_x_attn_cache_k_{index}")
            self.decoder_cached_input_names.append(f"input_x_attn_cache_v_{index}")

        self.decoder_output_names = ["logits"]
        for index in range(self.layer_count):
            self.decoder_output_names.append(f"output_cache_k_{index}")
            self.decoder_output_names.append(f"output_cache_v_{index}")
            self.decoder_output_names.append(f"output_x_attn_cache_k_{index}")
            self.decoder_output_names.append(f"output_x_attn_cache_v_{index}")
        
        if model_dir is None:
            decoder_initial, decoder, encoder, preprocessor = self._load_weights_from_hf_hub(
                model_name, model_precision
            )
        else:
            decoder_initial, decoder, encoder, preprocessor = [
                f"{model_dir}/{x}.tfl"
                for x in ("decoder_initial", "decoder", "encoder", "preprocessor")
            ]
 

        self.preprocessor = tflite.Interpreter(
            preprocessor, 
            # experimental_op_resolver_type=tflite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )
        self.encoder = tflite.Interpreter(
            encoder, 
            # experimental_op_resolver_type=tflite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )
        self.decoder_initial = tflite.Interpreter(
            decoder_initial, 
            # experimental_op_resolver_type=tflite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )
        self.decoder = tflite.Interpreter(
            decoder, 
            # experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )

        self.preprocessor.allocate_tensors()
        self.encoder.allocate_tensors()
        self.decoder_initial.allocate_tensors()
        self.decoder.allocate_tensors()

    def _load_weights_from_hf_hub(self, model_name, model_precision):
        model_name = model_name.split("/")[-1]
        return _get_tflite_weights(model_name, model_precision)

    def _invoke_tflite(self, interpreter, inputs, input_names=["input_000"], output_names=["output_000"]):
        input_details = interpreter.get_input_details()
        assert len(input_details) == len(input_names), f"Expected {len(input_names)} inputs, but got {len(input_details)}"
        output_details = interpreter.get_output_details()
        assert len(output_details) == len(output_names), f"Expected {len(output_names)} outputs, but got {len(output_details)}"

        input_indices = [None] * len(inputs)
        for input_index, input_name in enumerate(input_names):
            for input_detail in input_details:
                name_part = input_detail["name"].split(":")[0]
                if name_part == input_name:
                    input_indices[input_index] = input_detail["index"]
                    break
        assert None not in input_indices, f"Input names {input_names} not found in input details {input_details}"
        
        # Resize inputs and allocate tensors.
        for index, input in enumerate(inputs):
            interpreter.resize_tensor_input(input_indices[index], input.shape)
        interpreter.allocate_tensors()

        # Set inputs and invoke.
        for index, input in enumerate(inputs):
            interpreter.set_tensor(input_indices[index], input)
        interpreter.invoke()

        output_indices = [None] * len(output_names)
        for output_index, output_name in enumerate(output_names):
            for output_detail in output_details:
                name_part = output_detail["name"].split(":")[0]
                if name_part == output_name:
                    output_indices[output_index] = output_detail["index"]
                    break
        assert None not in output_indices, f"Output names {output_names} not found in output details {output_details}"

        return [interpreter.get_tensor(idx) for idx in output_indices]

    def generate(self, audio, max_len=228):
        features = self._invoke_tflite(self.preprocessor, [audio])[0]
        embeddings = self._invoke_tflite(
            self.encoder, 
            [features, np.array(features.shape[-2], dtype=np.int32)],
            ["input_000", "input_001"])[0]
        tokens = np.array([[1]], dtype=np.int32)
        output = tokens
        seq_len = np.array([1], dtype=np.int32)
        logits, *cache = self._invoke_tflite(
            self.decoder_initial, 
            [tokens, seq_len, embeddings], 
            self.decoder_uncached_input_names, 
            self.decoder_output_names)
        for _ in range(max_len):
            tokens = np.argmax(logits, axis=-1).astype(np.int32)
            output = np.concatenate([output, tokens], axis=-1)
            if tokens[0, 0] == 2:
                break
            seq_len = seq_len + 1
            logits, *cache = self._invoke_tflite(
                self.decoder, 
                [tokens, seq_len, embeddings] + cache, 
                self.decoder_cached_input_names, 
                self.decoder_output_names)

        return output
