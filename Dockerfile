FROM nvidia/cuda:12.0.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

# Install software-properties-common to add repositories
RUN apt update && apt install -y software-properties-common

# Add deadsnakes PPA for newer Python versions
RUN add-apt-repository ppa:deadsnakes/ppa

# Install Python (e.g., Python 3.9) and pip
RUN apt update && apt install -y python3.9 python3.9-distutils python3.9-venv python3-pip

# Update pip for Python 3.9
RUN python3.9 -m pip install --upgrade pip

# Set Python 3.9 as the default Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2

# Graph-Walkers dependencies
RUN apt install libpython3.9-dev -y

# install git
RUN apt install git -y

# Copy the contents of the random-walk directory
COPY . /random-walk

# Set the working directory in the container
WORKDIR /random-walk

# install dependencies
RUN bash install.sh

CMD ["/bin/bash"]
