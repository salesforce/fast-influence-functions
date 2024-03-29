# syntax = docker/dockerfile:1.0-experimental
FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime

# working directory
WORKDIR /workspace

# ---------------------------------------------
# Project-agnostic System Dependencies
# ---------------------------------------------
RUN \
    # Install System Dependencies
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        wget \
        unzip \
        psmisc \
        vim \
        git \
        ssh \
        curl \
        lshw \
        ubuntu-drivers-common \
        ca-certificates \
        libjpeg-dev \
        libpng-dev && \
    rm -rf /var/lib/apt/lists/* && \
    # Install NVIDIA Driver
    # https://www.linuxbabe.com/ubuntu/install-nvidia-driver-ubuntu-18-04
    # ubuntu-drivers autoinstall && \
    # https://serverfault.com/questions/227190/how-do-i-ask-apt-get-to-skip-any-interactive-post-install-configuration-steps
    # https://stackoverflow.com/questions/38165407/installing-lightdm-in-dockerfile-raises-interactive-keyboard-layout-menu
    # apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    #     nvidia-driver-440 && \
    # rm -rf /var/lib/apt/lists/* && \
    # Install NodeJS
    # https://github.com/nodesource/distributions/blob/master/README.md#deb
    curl -sL https://deb.nodesource.com/setup_12.x | bash - && \
    apt-get install -y nodejs

# ---------------------------------------------
# Project-specific System Dependencies
# ---------------------------------------------
RUN \
    # Install `graph_tool`
    # https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions#debian-ubuntu
    # and `cairo` https://cairographics.org/download/
    echo "deb [ arch=amd64 ] https://downloads.skewed.de/apt bionic main" >> /etc/apt/sources.list && \
    apt-key adv --keyserver keys.openpgp.org --recv-key 612DEFB798507F25 && \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3-graph-tool \
        libcairo2-dev && \
    rm -rf /var/lib/apt/lists/* && \
    # Link the directory to `graph_tool`, which is installed in a differet python path
    ln -s /usr/lib/python3/dist-packages/graph_tool/ /opt/conda/lib/python3.7/site-packages/graph_tool
    # Clone the Apex Module (this requires torch)
    # git clone https://github.com/NVIDIA/apex /workspace/apex && \
    # cd /workspace/apex && \
    # pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && \

# ---------------------------------------------
# Build Python depencies and utilize caching
# ---------------------------------------------
COPY ./fast-influence-functions/requirements.txt /workspace/fast-influence-functions/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /workspace/fast-influence-functions/requirements.txt && \
    # for binding/linking path from host machines
    mkdir -p /nlp && \
    mkdir -p /export/ && \
    chmod -R 777 /export

# upload everything
COPY ./fast-influence-functions/ /workspace/fast-influence-functions/

# Set HOME
ENV HOME="/workspace/fast-influence-functions"

# ---------------------------------------------
# Project-agnostic User-dependent Dependencies
# ---------------------------------------------
RUN \
    # Install FZF the fuzzy finder
    git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf && \
    ~/.fzf/install --all && \
    # Install Awesome vimrc
    git clone --depth=1 https://github.com/amix/vimrc.git ~/.vim_runtime && \
    sh ~/.vim_runtime/install_awesome_vimrc.sh

# Reset Entrypoint from Parent Images
# https://stackoverflow.com/questions/40122152/how-to-remove-entrypoint-from-parent-image-on-dockerfile/40122750
ENTRYPOINT []

# load bash
CMD /bin/bash