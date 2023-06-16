# TensorFlow Serving Demo

## 1. Directory Structure
```
multiModel/
├── model1
│   └── 00000123
│   　　 ├── saved_model.pb
│   　　 └── variables
│   　　　　 ├── variables.data-00000-of-00001
│   　　　　 └── variables.index
├── model2
│   　　└── 00000123
│   　　　　 ├── saved_model.pb
│   　　　　 └── variables
│   　　　　　　 ├── variables.data-00000-of-00001
│   　　　　　　 └── variables.index
└── models.config
```

## 2. models.config
```
model_config_list:{
    config:{
        name:"model1",
        base_path:"/models/multiModel/model1",
        model_platform:"tensorflow"
    },
    config:{
        name:"model2",
        base_path:"/models/multiModel/model2",
        model_platform:"tensorflow"
    },
}
```

## 3. Script
```bash
docker run -p 8501:8501 --mount type=bind,source=/home/node1/model/multiModel/,target=/models/multiModel -t tensorflow/serving:latest-gpu --model_config_file=/models/multiModel/models.config

docker run -it --rm --name yolo-tfserving -p 8501:8501 --gpus all -v "$(pwd)/models:/models" tensorflow/serving:latest-gpu --model_config_file=/models/models.config --allow_version_labels_for_unavailable_models=true
```

NOTE: `--allow_version_labels_for_unavailable_models=true` is used to allow version labels for unavailable models.