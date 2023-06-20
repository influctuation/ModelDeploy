# Triton Advanced Features

## Advanced configuration
### Version_Policy

If your model directory includes multiple versions, you should use `version_policy` to indicate which version to use, otherwise, the latest version will be used implictly.
Choose one of the following format:

```protobuf
version_policy: { all { } }
version_policy: {latest { num_versions: 1 }}
version_policy: { specific { versions: 1,2 } }
```

+ `All`: All versions of the model that are available in the model repository are available for inferencing.
+ `Latest`: Only the latest 'n' versions of the model in the repository are available for inferencing. The latest versions of the model are the numerically greatest version numbers.
+ `Specific`: Only the specifically listed versions of themodel are available for inferencing.

### Instance Group

`instance_group` Define a group of model instances running on the same device

+ `count`: The number of model instances on **each** device.
+ `kind` : The device type. `KIND_CPU` or `KIND_GPU`
+ `gpus` : The GPU device ID. If not specified, the model instance will be placed on the **each** available GPU device.

```protobuf
instance_group [
  {
    count: 2
    kind: KIND_CPU
  },
  {
    count: 1
    kind: KIND_GPU
    gpus: [0]
  },
  {
    count: 2
    kind: KIND_GPU
    gpus: [1,2]
  }
]
```

### Schedualing and Batching

#### Default schedualer
+ no batching
+ send requests as they are

#### Dynamic Batcher
+ combines requests as batch dynamically
+ key feature for increasing throughput
+ only for stateless models

arguments:
1. `preferred_batch_size` : the batch sizes that the dynamic batcher should attempt to create

2. `max_queue_delay_microseconds` :the time limit to wait for batching requests. 

Generally, the larger the `preferred_batch_size` and `max_queue_delay_microseconds`, the higher the throughput, but the longer the latency.

```protobuf
dynamic_batching {
  preferred_batch_size: [ 4，8 ]
  max_queue_delay_microseconds : 100
}
```

3. preserve ordering
4. priority levels
5. Queue Policy

#### Sequence Batchers
+ Used for stateful models
+ Ensures a sequence of inference requests are routed to the same modelinstance

#### Ensemble Schedualer
+ Only used for ensemble models

### Optimization Policy

#### Onnx with TensorRT Optimization(TRT backend for ONNX)
```protobuf
optimization { execution_accelerators { 
  gpu_execution_accelerator : [ {
    name : "tensorrt"
    parameters { 
      key: "precision_mode" 
      value: "FP16" 
    }
    parameters { 
      key : "max_workspace_size_bytes"
      value: "1073741824"
    }
  }]
}}
```

#### Tensorflow with TensorRT Optimization(TF-TRT)
```protobuf
optimization { execution_accelerators {
  gpu_execution_accelerator : [ {
    name : "tensorrt"
    parameters { 
      key: "precision_mode" 
      value: "FP16" 
    }
  }]
}}

```

### Model Warmup
+ Initialization may be deferred until the model receivesits first few inference requests
+ model_warmup makes Triton not show the model asready until warmup has completed
+ will cause Triton to be less responsive to model update

```protobuf
model_warmup [
  {
    batch_size: 64
    name: "warmup_requests"
    inputs {
      key: "input" 
      value: {
        random_data: true
        dims : [ 299，299，3]
        data_type: TYPE_FP32
      }
    }
  }
]
```

## Python backend and business logic script(bls)

For those logic that deep learning does not support or hard to implement, 
you can use Python backend to implement them. For example, 
pre/post-processing operations, dynamic condition/loop which are not supported by static ensemble models.

### How to use Python backend

1. Set the `backend` property in the config.pbtxt configuration file to `python`.

2. Install modules that are not included in the python standard library
   like `torch`,`torchvision`,`onnxrutime` and so on in the tritonserver container.
   Note that you should use pip3 instead of pip to install those packages.
   Specially, if you want to use `cv2`, you should install `opencv-python-headless` instead of `opencv-python`.

3. Create version directory and put `model.py` and `TritonPythonModel` class in it.
   You could only use `TritonPythonModel` as the name of the class to implement your business logic.

4. implement the `initialize(self, args)`,`execute(self, requests)`,`finalize(self)` method in the 
   `TritonPythonModel` class. You must implement the `execute` method, the other two methods are optional. For each received resquest, you must return a response.

NOTE: 

+ Even if you have specified to use GPU instance in `instance_group`, the python backend weil still 
copy them on CPU. To avoid copy overhead, you should set `FORCE_CPU_ONLY_INPUT_TENSORS` parameter
in config.pbtxt to `no`.
```protobuf
parameters: {
    key: "FORCE_CPU_ONLY_INPUT_TENSORS"
    value: { string_value: "no" }
}
```
+ The python backend only supports `python_backed_tensor` and `numpy` as the data type.If you are using `torch.Tensor` you should use `from_dlpack` and `to_dlpack` in `torch.utils.dlpack` to convert.