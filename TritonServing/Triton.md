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

# config.pbtxt file
## Fundamental configuration
### Platform | Backend 
||TensorRT|ONNX|TensorFlow|PyTorch|OpenVINO|Python|DALI|Custom|
|---|---|---|---|---|---|---|---|---|
|Platform|tensorrt_plan|onnxruntime_onnx|**tensorflow_graphdef** or **tensorflow_savedmodel**(required)|pytorch_libtorch|/|/|/|<del>custom_custom(optional)</del>|
|Backed|tensorrt|onnxruntime|tensorflow(optional)|pytorch|**openvino**(required)|**python**(required)|**dali**(required)|<del>custom</del>|

**Note:** 

If one is marked as `required`, then it must be present **explicitly** in the model configuration. If one is maked as `optional`, then it can be ommitted in the model configuration.

For `TensorRT`,`Onnx`, `Tensorflow saved_model` models, if you use `--strict-model-config=false`, then config.pbtxt is not required.

When you are using pytorch backend, please check whether your model is exported as `torchscript` format.

### Input/Output

For `pytorch` models the input/output name is must be `INPUT__*` and `OUTPUT__*`.(double underscore)

If `max_batch_size` is set to 0, then the input/output batch dim is excluded, other wise, the input/output batch dim is dynamic.

If your input shape is variable, you can set `dims` to `-1` to indicate that the input shape is variable.

If your input shape is not the same as the model input shape, you can add `reshape` property to indicate.
```protobuf
dims :[3,244,244]
reshape { shape : [1,3,244,244] }
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

 # Lunch Triton Server
 
 ## Docker CLI Command
 ```bash
docker run -it --gpus all --name TritonServer -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $pwd/models:/models nvcr.io/nvidia/tritonserver:23.05-py3 tritonserver --model-repository=/models
 ```

 ## tritonserver options
+ `--log-verbose <integer>`
  
  Set verbose logging level. Zero (0) disables verbose loggingand values >= 1 enable verbose logging.

+ `--strict-model-config <boolean>`

  lf true model configuration files must be provided and all required configuration settings must be specified.

  lf false the model configuration may be absent or only
  partially specified and the server will attempt to derive the missing required configuration.

+ `--strict-readiness <boolean>`

  lf true, the server is responsive only when all models are ready.

  lf false, the server is responsive even if somelall models areunavailable.
  
+ `--exit-on-error <boolean>`

  lf false, even when some of models fail to be loaded theserver will still be launched.

+ `--http-port <integer>`

  The port for the server to listen on for HTTP requests.

+ `--grpc-port <integer>`

  The port for the server to listen on for GRPC requests.

+ `--metrics-port <integer>`

  The port reporting prometheus metrics.

+ `--model-control-mode <string>`

  Specify the mode for model management.Options are **"none"**,**"poll"** and **"explicit"**.

  `None`(default) for **static** model loading, `Poll` for polling the model repository for changes, `Explicit` for loading models explicitly via the **model control APIs**.

+ `--repository-poll-secs <integer>`

  Interval in seconds between each poll of the model repositoryto check for changes. Valid only when `--model-control-mode=poll` is specified.

+ `--load-model <string>`

  Name of the model to be loaded on server startup.O only take affect if `--model-control-mode=explicit` is true.

+ `--pinned-memory-pool-byte-size <integer>`

  total byte size that can be allocated as pinned system memory which is used for accelerating data transfer between host and devices.Default is 256M.

+ `--cuda-memory-pool-byte-size <<integer>:<integer>>`

  total byte size that can be allocated as CUDA memory for the GPU device. Default is 64M.--backend-directory <string>
  The global directory searched for backend shared libraries.Default is 'lopt/tritonserver/ backends'.

+ `--repoagent-directory <string>`

  The global directory searched for repository agent sharedlibraries.Default is 'lopt/tritonserver/repoagents'.
