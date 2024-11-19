FROM nvcr.io/nvidia/doca/doca:2.8.0-devel-cuda12.1.0-host

# apt update
RUN apt-get update
# install the gpunetio lib
RUN apt install doca-all doca-sdk-gpunetio libdoca-sdk-gpunetio-dev -y
# set some enviroment variables

