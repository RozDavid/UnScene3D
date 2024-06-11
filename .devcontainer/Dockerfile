FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    nano \
    tree \
    unzip \
    wget \
    python-is-python3 \
    python3-pip \
    libopenblas-dev \
    ninja-build \
    # Open3D
    libx11-6 \
    libgl1-mesa-glx \
    # OpenCV
    libglib2.0-dev \
    # Rendering and videos 
    ffmpeg \
    # Custom builds
    cmake build-essential libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Pretty terminal colors
ENV TERM xterm-256color

# So the user is not faced with an empty list of apt packages
RUN apt-get update
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# Copy the environment file and install 
COPY conf/unscene3d_requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt  \
    --find-links https://download.pytorch.org/whl/torch \ 
    --find-links https://download.pytorch.org/whl/torchvision \
    && rm /tmp/requirements.txt

# Install pybind 11 and torch scatter only after 
RUN pip install torch_scatter -f https://data.pyg.org/whl/torch-1.13.0+cu116.html && pip install pybind11 open3d volumentations

# # Install other modules 
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6+PTX"
ENV MAX_JOBS=8
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Install minkowski engine
RUN git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine" tmp/MinkowskiEngine
RUN cd tmp/MinkowskiEngine; python setup.py install --force_cuda --blas=openblas

# # Install detectron2
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# # Build internal libraries
COPY third_party/pointnet2 /tmp/pointnet2
RUN cd /tmp/pointnet2 && python setup.py install && cd ../..
COPY utils /tmp/utils
RUN cd /tmp/utils/cpp_utils && python setup.py install && \
    cd /tmp/utils/cuda_utils && python setup.py install 

# Install pseudo mask related components
RUN pip install clip hdbscan hydra-submitit-launcher

# Ensures file permissions are consistent
ARG USER_ID
ARG GROUP_ID
RUN groupadd --gid $GROUP_ID user || echo "Group already exists"
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

USER user
WORKDIR /UnScene3D
