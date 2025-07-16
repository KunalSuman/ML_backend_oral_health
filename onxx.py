import tensorflow as tf
import tf2onnx

model = tf.keras.models.load_model("oral_classifier_20250715_2139.h5")
if not hasattr(model, "output_names"):
    # Set to the name of the output tensor
    model.output_names = [model.output.name.split(":")[0]]

spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
output_path = "oral_classifier.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)
