from moonshine import load_model
import os
import tensorflow as tf
from tensorflow.python.tools import saved_model_utils

def save_tfl(keras_model, path, input_names=None, output_names=None):
    saved_model_path = path.replace(".tfl", ".saved_model")
    # print(keras_model.summary())
    # print(keras_model.inputs)
    # exit(1)
    keras_model.export(saved_model_path, format="tf")

    tag_set = ["serve"]
    signature_def_key = 'serving_default'

    saved_model = saved_model_utils.read_saved_model(saved_model_path)
    for meta_graph_def in saved_model.meta_graphs:
        if meta_graph_def.meta_info_def.tags != tag_set:
            continue
        signature_def = meta_graph_def.signature_def[signature_def_key]

        if input_names is not None:
            assert len(input_names) == len(signature_def.inputs), (
                f"input_names ({input_names}, len: {len(input_names)}) must have the same length as the number of inputs ({len(signature_def.inputs)})"
            )
        for index, (_, info) in enumerate(signature_def.inputs.items()):
            input_node_name, input_port = info.name.split(":")
            if input_names is not None:
                new_node_name = input_names[index]
            else:
                new_node_name = f"input_{int(index):03d}"
            for node in meta_graph_def.graph_def.node:
                if node.name == input_node_name:
                    node.name = new_node_name
                for i, input in enumerate(node.input):
                    if input == input_node_name:
                        node.input[i] = f"{new_node_name}:{input_port}"
                        break                    
            info.name = f"{new_node_name}:{input_port}"

        if output_names is not None:
            assert len(output_names) == len(signature_def.outputs), (
                "output_names must have the same length as the number of outputs"
            )
        for index, (_, info) in enumerate(signature_def.outputs.items()):
            output_node_name, output_node_index = info.name.split(":")
            if output_names is not None:
                new_node_name = output_names[index]
            else:
                new_node_name = f"output_{int(index):03d}"
            meta_graph_def.graph_def.node.extend([
                tf.compat.v1.NodeDef(
                    name=new_node_name, op="Identity", 
                    input=[output_node_name + ":" + str(output_node_index)], 
                    attr={"T": tf.compat.v1.AttrValue(type=info.dtype)},
                    device="")
            ])
            info.name = f"{new_node_name}:0"
            
            # for node in meta_graph_def.graph_def.node:
            #     print(f"Node: {node.name}, {node.op}, {node.input}, {node.attr["T"]}")
            # exit(0)
            
    saved_model_pb_path = os.path.join(saved_model_path, "saved_model.pb")

    # Preserve the original meta graph for debugging purposes.
    original_model_pb_path = os.path.join(saved_model_path, "saved_model_original.pb")
    if os.path.exists(original_model_pb_path):
        os.remove(original_model_pb_path)
    os.rename(saved_model_pb_path, original_model_pb_path)
    
    with open(saved_model_pb_path, "wb") as file:
        file.write(saved_model.SerializeToString())
        
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(tflite_model)

# for model_name in ["tiny", "base"]:
for model_name in ["tiny"]:
    
    model = load_model(f"moonshine/{model_name}")

    precision = "float"

    preprocessor_keras = model.preprocessor.preprocess
    encoder_keras = model.encoder.encoder
    decoder_uncached_keras = model.decoder.uncached_call
    decoder_cached_keras = model.decoder.cached_call

    assert model_name in ["tiny", "base"]
    if model_name == "tiny":
        layer_count = 6
    elif model_name == "base":
        layer_count = 8

    decoder_uncached_input_names = [
        "tokens",
        "audio_features",
        "seq_len",
    ]

    decoder_cached_input_names = decoder_uncached_input_names.copy()
    for index in range(layer_count):
        decoder_cached_input_names.append(f"input_cache_k_{index}")
        decoder_cached_input_names.append(f"input_cache_v_{index}")
        decoder_cached_input_names.append(f"input_x_attn_cache_k_{index}")
        decoder_cached_input_names.append(f"input_x_attn_cache_v_{index}")

    decoder_output_names = ["logits"]
    for index in range(layer_count):
        decoder_output_names.append(f"output_cache_k_{index}")
        decoder_output_names.append(f"output_cache_v_{index}")
        decoder_output_names.append(f"output_x_attn_cache_k_{index}")
        decoder_output_names.append(f"output_x_attn_cache_v_{index}")
        
    save_tfl(preprocessor_keras, os.path.join(model_name, precision, "preprocessor.tfl"))
    save_tfl(encoder_keras, os.path.join(model_name, precision, "encoder.tfl"))
    save_tfl(
        decoder_uncached_keras, 
        os.path.join(model_name, precision, "decoder_initial.tfl"),
        input_names=decoder_uncached_input_names, 
        output_names=decoder_output_names)
    save_tfl(
        decoder_cached_keras, 
        os.path.join(model_name, precision, "decoder.tfl"),
        input_names=decoder_cached_input_names, 
        output_names=decoder_output_names)
