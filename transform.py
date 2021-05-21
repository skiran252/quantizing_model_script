import torch
import numpy as np
from simpletransformers.model import TransformerModel
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import onnx
import onnxruntime

model = TransformerModel(
    "roberta", "Roberta_evaluator_6", args=({"fp16": False}), num_labels=2,use_cuda=False
)


tokenizer = RobertaTokenizer.from_pretrained("Roberta_evaluator_6")

question = "who is the father of the nation?"
answer = "Mahatma Gandhi"

input_data = tokenizer(
    text=question,
    text_pair=answer,
    padding="max_length",
    max_length=512,
    truncation=True,
    return_tensors="pt",
)

torch.onnx.export(
    model.model,
    (input_data["input_ids"], input_data["attention_mask"]),
    "roberta.onnx",
    opset_version=11,
    input_names=["input_ids", "attention_mask"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size", 1: "sentence_length"},
        "output": {0: "batch_size"},
    },
)