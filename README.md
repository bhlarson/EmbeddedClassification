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


1. Stop and remove the current cl
   > docker container ls
   note the IMAGE ID for cl
    > docker stop <CONTAINER ID>

1. Load docker image
   > docker load -i containers/cl.tar 
1. Run docker image
   > docker run --rm -it --volume "$(pwd):/app/ec" --volume "/store/Datasets/imdb:/trainingset" --gpus '"device=0"' -p 6006:6006/tcp -p 3000:3000 ec:latest
1.  
   > docker run --rm -it --volume "$(pwd):/app/ec" --volume "/store/Datasets/imdb:/trainingset" --gpus '"device=0"' -p 6006:6006/tcp ec:latest