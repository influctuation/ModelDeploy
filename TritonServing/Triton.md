## 目录结构
``` 
  <model-repository-path>/# 模型仓库目录
    <model-name>/ # 模型名字
      [config.pbtxt] # 模型配置文件
      [<output-labels-file> ...] # 标签文件，可以没有
      <version>/ # 该版本下的模型
        <model-definition-file>
      <version>/
        <model-definition-file>
      ...
    <model-name>/
      [config.pbtxt]
      [<output-labels-file> ...]
      <version>/
        <model-definition-file>
      <version>/
        <model-definition-file>
      ...
    ...
```

# .config file
## Fundamental configuration
### Platform | Backend 
||TensorRT|ONNX|TensorFlow|PyTorch|OpenVINO|Python|DALI|Custom|
|---|---|---|---|---|---|---|---|---|
|Platform|tensorrt_plan|onnxruntime_onnx|**tensorflow_graphdef** or **tensorflow_savedmodel**(required)|pytorch_libtorch|/|/|/|<del>custom_custom(optional)</del>|
|Backed|tensorrt|onnxruntime|tensorflow(optional)|pytorch|**openvino**(required)|**python**(required)|**dali**(required)|<del>custom</del>|

**Note:** 

If one is marked as `required`, then it must be present **explicitly** in the model configuration. If one is maked as `optional`, then it can be ommitted in the model configuration.

For `TensorRT`,`Onnx`, `Tensorflow saved_model` models, if you use `--strict-model-config=false`, then config.pbtxt is not required.

### Input/Output

For `pytorch` models the input/output name is must be `INPUT__*` and `OUTPUT__*`.(double underscore)

If `max_batch_size` is set to 0, then the input/output batch dim is excluded, other wise, the input/output batch dim is dynamic.

If your input shape is variable, you can set `dims` to `-1` to indicate that the input shape is variable.

If your input shape is not the same as the model input shape, you can add `reshape` propertt to indicate.
```protobuf
dims :[3,244,244]
reshape { shape : [1,3,244,244] }
```

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

## Protocol
Triton exposes both HTTP/REST and GRPC endpoints based on standard inference protocols that have been proposed by the KServe project.

REST/HTTP:[kserve/REST](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md)
GRPC:[kserve/GRPC](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/grpc_predict_v2.proto)

**Health:**

 GET `v2/health/live`

 GET `v2/health/ready`

 GET `v2/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]/ready`

**Server Metadata:**

 GET `v2`

**Model Metadata:**

 GET `v2/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]`

**Inference:**

 POST `v2/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]/infer`

 # Docker command to lunch Triton Server
 
 ```bash
docker run -it --gpus all --name TritonServer -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $pwd/models:/models nvcr.io/nvidia/tritonserver:23.05-py3 tritonserver --model-repository=/models
 ```