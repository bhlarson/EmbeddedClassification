FROM tensorflow/tensorflow:2.1.0-gpu-py3

LABEL maintainer="Brad Larson"
#COPY README.md /app/README.md
#COPY fcn /app/fcn/
#COPY Model_zoo /app/Model_zoo/

RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip3 install --upgrade pip
RUN pip3 --no-cache-dir install \
        opencv-python==4.2.0.34 \
        scipy==1.4.1 \
        numpy==1.18.2 \
        Pillow==7.1.1 \
        minio==5.0.10 \
        natsort==7.0.1 \
        ptvsd==4.3.2 \
        matplotlib==3.2.1 \
        datetime==4.3 \
        flask==1.1.1 \
        tf2onnx

WORKDIR /app
ENV LANG C.UTF-8
# port 6006 exposes tensorboard
EXPOSE 6006 
# port 3000 exposes debugger
EXPOSE 3000

# Launch training
#ENTRYPOINT ["python", "train_imdb.py"]
# Launch bash shell
CMD ["/bin/bash"]