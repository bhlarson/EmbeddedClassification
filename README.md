# EmbeddedClassification
Embedded image classification based on tensorflow model
Resnet model trained on IMDB dataset to classify image age and gender

Dataset: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/

1. Build development docker image
> docker build --pull --rm -f "dockerfile" -t ec:latest .

1. Run development docker image
   > docker run --gpus '"device=0"' -it --rm -v "$(pwd):/app" -v "/store:/store" -p 6006:6006/tcp -p 5000:5000/tcp -p 3000:3000 ec:latest

1. Run runtime docker image
   > 

1. In docker container, convert dataset into record:
   > python makecrecord.py
1. In docker container, train network:
   > python train_imdbv2.py
1. In docker container, test network:
   > python test_imdb.py # Not yet updated to TF 2
1. In docker container, evaluate images:
   > python infer_imdb.py
1. Test on Triton server
   > docker pull nvcr.io/nvidia/tritonserver:20.03-py3
   > docker run -it --rm --gpus device=0 --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p 8000:8000 -p 8001:8001 -p 8002:8002 -v"/data/models/inference/trtis/lit:/models/lit" nvcr.io/nvidia/tritonserver:20.03-py3 trtserver --model-repository=/models
   > docker run -it --rm --gpus device=0 --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p 8000:8000 -p 8001:8001 -p 8002:8002 trtis:latest trtserver --model-repository=s3://192.168.1.66:19002/models

1. Target Jetson:

## Test on Google Corel Board:
1. Set up Coral [Edge TPU](https://coral.ai/docs/dev-board/get-started/)
1. Train model with TensorFlow 1.15 until Edge TPU compiler supports Tensorflow 2.0
1. Quantize trained model to int8 [](https://www.tensorflow.org/lite/convert/python_api)
1. [Quatization aware training](https://github.com/tensorflow/tensorflow/tree/r1.13/tensorflow/contrib/quantize)





Docker tensorrt
> docker build --rm -f imgdfn/dockerfile -t trt01:latest imgdfn/context

Run TRT image:
> docker run --gpus '"device=0"' -it --rm -v "$(pwd):/app/ec" -v "/store/Datasets/imdb:/trainingset" -v "/store/Datasets:/store/Datasets" -p 6006:6006/tcp -p 3000:3000 trt:latest

> docker run --gpus '"device=0"' -it --rm -v "$(pwd):/app/ec" -v "/store/Datasets/imdb:/trainingset" -v "/store/Datasets:/store/Datasets" -p 6006:6006/tcp -p 3000:3000 nvcr.io/nvidia/tensorrt:20.03-py3

> docker run --gpus '"device=0"' -it --rm -v "$(pwd):/app" -v "/store/Datasets/imdb:/trainingset" -v "/store/Datasets:/store/Datasets" -p 3000:3000 trt01:latest



python -m tensorflow.python.tools.freeze_graph --input_saved_model_dir SAVED_MODEL_DIR