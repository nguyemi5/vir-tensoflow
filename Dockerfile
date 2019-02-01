FROM ubuntu:16.04
# LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"
# 
# RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates apt-transport-https gnupg-curl && \
#     rm -rf /var/lib/apt/lists/* && \
#     NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 && \
#     NVIDIA_GPGKEY_FPR=ae09fe4bbd223a84b2ccfce3f60f4b3d7fa2af80 && \
#     apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub && \
#     apt-key adv --export --no-emit-version -a $NVIDIA_GPGKEY_FPR | tail -n +5 > cudasign.pub && \
#     echo "$NVIDIA_GPGKEY_SUM  cudasign.pub" | sha256sum -c --strict - && rm cudasign.pub && \
#     echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list
# 
# ENV CUDA_VERSION 8.0.61
# 
# ENV CUDA_PKG_VERSION 8-0=$CUDA_VERSION-1
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         cuda-nvrtc-$CUDA_PKG_VERSION \
#         cuda-nvgraph-$CUDA_PKG_VERSION \
#         cuda-cusolver-$CUDA_PKG_VERSION \
#         cuda-cublas-8-0=8.0.61.2-1 \
#         cuda-cufft-$CUDA_PKG_VERSION \
#         cuda-curand-$CUDA_PKG_VERSION \
#         cuda-cusparse-$CUDA_PKG_VERSION \
#         cuda-npp-$CUDA_PKG_VERSION \
#         cuda-cudart-$CUDA_PKG_VERSION && \
#     ln -s cuda-8.0 /usr/local/cuda && \
#     rm -rf /var/lib/apt/lists/*
# 
# # nvidia-docker 1.0
# LABEL com.nvidia.volumes.needed="nvidia_driver"
# LABEL com.nvidia.cuda.version="${CUDA_VERSION}"
# 
# RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
#     echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf
# 
# ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
# ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
# 
# # nvidia-container-runtime
# ENV NVIDIA_VISIBLE_DEVICES all
# ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
# ENV NVIDIA_REQUIRE_CUDA "cuda>=8.0"
# 
# RUN echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
# 
# ENV CUDNN_VERSION 6.0.21
# LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
# 
# RUN apt-get update && apt-get install -y --no-install-recommends \
#             libcudnn6=$CUDNN_VERSION-1+cuda8.0 && \
#     rm -rf /var/lib/apt/lists/*

# configure environment
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH
ENV CONTAINER_UID 1000
ENV INSTALLER Miniconda2-latest-Linux-x86_64.sh

RUN apt-get update && apt-get install --no-install-recommends -y wget

# create conda directory for lion user
RUN mkdir -p /opt/conda

# install conda with python 2.7
RUN cd /tmp && \
mkdir -p $CONDA_DIR && \
wget --no-check-certificate https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh && \
echo $(wget --quiet -O - https://repo.continuum.io/miniconda/ \
| grep -A3 $INSTALLER \
| tail -n1 \
| cut -d\> -f2 \
| cut -d\< -f1 ) $INSTALLER | md5sum -c - && \
/bin/bash $INSTALLER -f -b -p $CONDA_DIR && \
rm $INSTALLER

RUN conda install -y tensorflow-gpu==1.4.0

# install packages
RUN apt-get update && apt-get install -q -y \
    dirmngr \
    gnupg2 \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 421C365BD9FF1F717815A3895523BAEEB01FA116

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" > /etc/apt/sources.list.d/ros-latest.list

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    python-rosdep \
    python-rosinstall \
    python-vcstools \
    && rm -rf /var/lib/apt/lists/*

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# bootstrap rosdep
RUN rosdep init \
    && rosdep update

# install ros packages
ENV ROS_DISTRO kinetic
RUN apt-get update && apt-get install -y \
    ros-kinetic-ros-core=1.3.2-0* \
    && rm -rf /var/lib/apt/lists/*

# install ros packages
RUN apt-get update && apt-get install -y \
    ros-kinetic-desktop-full=1.3.2-0* \
    && rm -rf /var/lib/apt/lists/*
    
RUN apt-get update && apt-get install -y \
    libalglib-dev \
    gnuplot \
    python-pip \
    python-flufl.lock \
    python-opencv
    
RUN python -m pip install \
	h5py \
	scipy \
	parse \
	tqdm