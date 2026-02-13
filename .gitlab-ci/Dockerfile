FROM quay.io/pypa/manylinux2014_x86_64:2025.10.10-1

# install additional vcpkg and vortex build dependencies
RUN yum -y install zip unzip elfutils-devel perl-IPC-Cmd autoconf-archive

# install numpy for the default Python so that xtensor-python builds
RUN /usr/local/bin/python3.14 -m pip install numpy

# build newer Ninja than ninja-build package provides (vcpkg downloads a newer one that requires GLIBCXX_3.4.20 not available here)
RUN git clone https://github.com/ninja-build/ninja && cd ninja && python3.14 configure.py --bootstrap && cp ninja /usr/local/bin && cd .. && rm -rf ninja

# downgrade to CMake 3.31
# RUN curl -LO https://github.com/Kitware/CMake/releases/download/v3.31.10/cmake-3.31.10-linux-x86_64.sh && sh cmake-3.31.10-linux-x86_64.sh --skip-license --prefix=/usr/local && rm cmake-3.31.10-linux-x86_64.sh

# fix environment variables
ENV PATH="/usr/local/cuda/bin:$PATH"
