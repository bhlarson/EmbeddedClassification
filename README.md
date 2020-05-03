# EmbeddedClassification
Embedded image classification based on tensorflow model

Build training docker image and add to kubernetes registry:
> docker build --rm -f "dockerfile_fcntrain" -t localhost:32000/cl:latest "."
Push image into registry:
> docker push localhost:32000/cl

view registry in browser through the URL:
http://192.168.1.66:32000/v2/_catalog 

log in to kubernetes dashboard:

https://192.168.1.66:32090/#!/login


1. Build docker image
> docker build --rm -f dockerfile -t ec:latest context

1. Stop and remove the current cl
   > docker container ls
   note the IMAGE ID for cl
    > docker stop <CONTAINER ID>

1. Load docker image
   > docker load -i containers/cl.tar 
1. Run docker image
   > docker run --rm -it -v "$(pwd):/app" -v "/store/Datasets/imdb:/trainingset" -v "/store/Datasets:/store/Datasets" --gpus '"device=0"' -p 6006:6006/tcp -p 3000:3000 ec:latest
1. To increase logging of TRT conversion, launch python with the desired logging levels
   > TF_CPP_VMODULE=segment=1,convert_graph=1,convert_nodes=1,trt_engine_op=1 python savemodelv2.py > conversion.txt 2>&1
1.  
   > docker run --rm -it -v "$(pwd):/app" -v "/store/Datasets/imdb:/trainingset" -v "/store/Datasets:/store/Datasets" --gpus '"device=0"' -p 6006:6006/tcp ec:latest

> docker run --gpus '"device=0"' -it --rm -v "$(pwd):/app/ec" -v "/store/Datasets/imdb:/trainingset" -v "/store/Datasets:/store/Datasets" -p 6006:6006/tcp -p 3000:3000 nvcr.io/nvidia/tensorrt:20.03-py3

Docker tensorrt
> docker build --rm -f imgdfn/dockerfile -t trt01:latest imgdfn/context

Run TRT image:
> docker run --gpus '"device=0"' -it --rm -v "$(pwd):/app/ec" -v "/store/Datasets/imdb:/trainingset" -v "/store/Datasets:/store/Datasets" -p 6006:6006/tcp -p 3000:3000 trt:latest

> docker run --gpus '"device=0"' -it --rm -v "$(pwd):/app/ec" -v "/store/Datasets/imdb:/trainingset" -v "/store/Datasets:/store/Datasets" -p 6006:6006/tcp -p 3000:3000 nvcr.io/nvidia/tensorrt:20.03-py3

> docker run --gpus '"device=0"' -it --rm -v "$(pwd):/app" -v "/store/Datasets/imdb:/trainingset" -v "/store/Datasets:/store/Datasets" -p 3000:3000 trt01:latest

> docker run --gpus '"device=0"' -it --rm -v "$(pwd):/app" -v "/store:/store" -p 6006:6006/tcp -p 3000:3000 ec:latest

python -m tensorflow.python.tools.freeze_graph --input_saved_model_dir SAVED_MODEL_DIR