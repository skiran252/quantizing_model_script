import torch
import numpy as np
from transformers import AutoTokenizer
import onnxruntime as rt
import onnxruntime
import multiprocessing
import numpy as np

sess_options = onnxruntime.SessionOptions()
# Set graph optimization level to ORT_ENABLE_EXTENDED to enable bert optimization.
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        # Use OpenMP optimizations. Only useful for CPU, has little impact for GPUs.
sess_options.intra_op_num_threads = multiprocessing.cpu_count()
SEQ_LENGTH = 512
QAE_PRETRAINED="Roberta_evaluator_6"
qae_tokenizer = AutoTokenizer.from_pretrained(QAE_PRETRAINED)
question="who is the father of the nation?"
answer = "gandhi"
input_data  = qae_tokenizer(
            text=question,
            text_pair=answer,
            padding="max_length",
            max_length=SEQ_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
input_data_onnx = {
    "input_ids":input_data["input_ids"].cpu().numpy(),
    "attention_mask":input_data["attention_mask"].cpu().numpy()
}
import time

sess = rt.InferenceSession("model.quant.onnx",sess_options)
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
for i in range(5):
    start = time.time()
    pred = sess.run(None, input_data_onnx)
    print("took",time.time()-start)
    print(pred[0])
    print(np.argmax(pred[0]))