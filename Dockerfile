FROM nvcr.io/nvidia/isaac-lab:2.0.2

WORKDIR /workspace

RUN git clone https://github.com/facebookresearch/r3m.git && cd r3m && /workspace/isaaclab/isaaclab.sh -p -m pip install -e .
RUN git clone https://github.com/StriderOne/IsaacStressor.git && cd IsaacStressor && /workspace/isaaclab/isaaclab.sh -p -m pip install -e source/isaac_stressor

RUN apt-get update && apt-get install vim -y