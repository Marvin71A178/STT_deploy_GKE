FROM storyteller123/cuda118_cudnn_pytorch210_simpletransformer
COPY . /deploy_Mood/



RUN python3 -m pip install --upgrade pip
RUN apt update
RUN apt install ffmpeg
RUN git clone https://github.com/facebookresearch/audiocraft.git

WORKDIR /deploy_Mood/