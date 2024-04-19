FROM storyteller123/cuda118_cudnn_pytorch210_simpletransformer
COPY . /deploy_Mood/
WORKDIR /deploy_Mood/
EXPOSE 8080

RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip install dora-search

RUN pip3 install torch==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3 install torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
RUN apt-get update && \
    apt-get install -y ffmpeg 


ENTRYPOINT [ "python3", "app.py" ]
# FROM test_deploy_gke4_bash
# COPY . /deploy_Mood2/
# WORKDIR /deploy_Mood2/
# ENTRYPOINT [ "python3", "app.py" ]