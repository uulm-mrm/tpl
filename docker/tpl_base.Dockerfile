FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ARG ARCHITECTURE

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y -qq --no-install-recommends \
        build-essential \
        sudo \
        libglfw3-dev \
        libglew-dev \
        mesa-utils \
        python3-dev \
        python3-pip \
        libeigen3-dev \
        liblapacke-dev \
        libyaml-cpp-dev \
        vim \
        git \
        zsh \
        tmux \
        wget \
        && rm -rf /var/lib/apt/lists/*

RUN wget "https://github.com/Kitware/CMake/releases/download/v3.29.2/cmake-3.29.2-Linux-$ARCHITECTURE.sh" \
      -q -O /tmp/cmake-install.sh \
      && chmod u+x /tmp/cmake-install.sh \
      && mkdir /opt/cmake-3.29.2 \
      && /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.29.2 \
      && rm /tmp/cmake-install.sh \
      && ln -s /opt/cmake-3.29.2/bin/* /usr/local/bin

RUN python3 -m pip install --upgrade pip==24.0
RUN python3 -m pip install \
    git+https://github.com/uulm-mrm/imdash@f9dfd2495eba9b88fa2ec1d0af4540f40dc6895a \
    'structstore==0.1.9' \
    'objtoolbox==0.0.10' \
    'imviz==0.2.7' \
    matplotlib \
    ranger-fm \
    pytest \
    parameterized

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

COPY ./docker/zshrc /etc/zsh/zshrc
COPY ./docker/tmux.conf /etc/tmux.conf
