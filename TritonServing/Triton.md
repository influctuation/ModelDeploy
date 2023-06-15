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