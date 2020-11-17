from onnxconverter_common.data_types import FloatTensorType
from onnxmltools.convert import convert_xgboost
import onnxmltools
import mxnet.contrib.onnx as onnx_mxnet

print ("running")
import sklearn, pickle
model = pickle.load(open("xgboost-model", "rb"))
print(model)

initial_type = [('float_input', FloatTensorType([None, 4]))]
onx = convert_xgboost(model, initial_types=initial_type)

# Save as protobuf
onnxmltools.utils.save_model(onx, 'example.onnx')

print("done")