from jaxonnxruntime import backend as jax_backend
import onnx
import jax.numpy as jp


class OnnxInfer:
    def __init__(self, onnx_model_path, input_name="obs"):
        self.input_name = input_name

        onnx_model = onnx.load(onnx_model_path)
        self.backend_rep = jax_backend.BackendRep(onnx_model)

    def infer(self, inputs):
        inputs = jp.array(inputs)
        output = self.backend_rep.run({self.input_name: jp.array([inputs])})
        return output[0][0]
