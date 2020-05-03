# EmbeddedClassification
Embedded image classification based on tensorflow model
Resnet model trained on IMDB dataset to classify image age and gender

Dataset: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/

1. Build docker image
> docker build --rm -f dockerfile -t ec:latest context

1. Run docker image
   > docker run --gpus '"device=0"' -it --rm -v "$(pwd):/app" -v "/store:/store" -p 6006:6006/tcp -p 3000:3000 ec:latest

1. In docker container, convert dataset into record:
   > python makecrecord.py
1. In docker container, train network:
   > python train_imdbv2.py
1. In docker container, test network:
   > python test_imdb.py # Not yet updated to TF 2
1. In docker container, evaluate images:
   > python infer_imdb.py




Docker tensorrt
> docker build --rm -f imgdfn/dockerfile -t trt01:latest imgdfn/context

Run TRT image:
> docker run --gpus '"device=0"' -it --rm -v "$(pwd):/app/ec" -v "/store/Datasets/imdb:/trainingset" -v "/store/Datasets:/store/Datasets" -p 6006:6006/tcp -p 3000:3000 trt:latest

> docker run --gpus '"device=0"' -it --rm -v "$(pwd):/app/ec" -v "/store/Datasets/imdb:/trainingset" -v "/store/Datasets:/store/Datasets" -p 6006:6006/tcp -p 3000:3000 nvcr.io/nvidia/tensorrt:20.03-py3

> docker run --gpus '"device=0"' -it --rm -v "$(pwd):/app" -v "/store/Datasets/imdb:/trainingset" -v "/store/Datasets:/store/Datasets" -p 3000:3000 trt01:latest



python -m tensorflow.python.tools.freeze_graph --input_saved_model_dir SAVED_MODEL_DIR