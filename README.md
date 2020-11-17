# onnxToDjl

The file "xgboost-model" was in a model.tar.gz file from a SageMaker training of the classic iris identification model (to perceived 100% accuracy, if that makes a difference).

The file "converter.py" reads this model and saves it as an ONNX model.

The file "ThirdApp.java" reads in the ONNX model - make sure to adjust the path to match where you have the ONNX file created above - and attempts to make a prediction.  This shows the point where ONNX can't load it.

The article with instructions for this is at https://docs.djl.ai/jupyter/onnxruntime/machine_learning_with_ONNXRuntime.html  I have followed the instructions for the inference, but of course the training was done on SageMaker.