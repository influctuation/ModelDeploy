# Model Deployment Guide Collections

## Why you should use Inference Engine rather than DeepLearning Framework

Inference engines like `Onnxrutime` and `TensorRT` are optimized for inference, which is the most important part of the whole deep learning pipeline. It is designed to be fast and efficient, and it is not designed to be flexible and easy to use. Deep learning frameworks such as `Pytorch` are designed to be flexible and easy to use, and they are not designed to be fast and efficient. So, if you want to deploy your model in production, you should use inference engine rather than deep learning framework.

## Popular Integrate Serving Frameworks

### TF Serving

[TensorFlow Serving](https://tensorflow.google.cn/tfx/guide/serving) is a flexible, high-performance serving system for machine learning models, designed for production environments. TensorFlow Serving makes it easy to deploy new algorithms and experiments, while keeping the same server architecture and APIs. TensorFlow Serving provides out-of-the-box integration with TensorFlow models, but can be easily extended to serve other types of models and data.

### Torch Serving

[TorchServe](https://pytorch.org/serve/) is a performant, flexible and easy to use tool for serving PyTorch models in production.

### Triton Serving

[NVIDIA Triton™](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html), an open-source inference serving software, standardizes AI model deployment and execution and delivers fast and scalable AI in production. Triton is part of NVIDIA AI Enterprise, an NVIDIA software platform that accelerates the data science pipeline and streamlines the development and deployment of production AI.

Triton enables teams to deploy any AI model from multiple deep learning and machine learning frameworks, including `TensorRT`, `TensorFlow`, `PyTorch`, `ONNX`, `OpenVINO`, Python, RAPIDS FIL, and more. Triton supports inference across cloud, data center,edge and embedded devices on NVIDIA GPUs, x86 and ARM CPU, or AWS Inferentia. Triton delivers optimized performance for many query types, including real time, batched, ensembles and audio/video streaming.
