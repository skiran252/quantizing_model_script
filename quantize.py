import onnx
from onnxruntime.quantization import quantize_qat, QuantType

model_fp32 = 'roberta.onnx'
model_quant = 'model.quant.onnx'
quantized_model = quantize_qat(model_fp32, model_quant)