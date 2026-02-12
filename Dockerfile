FROM tensorflow/tensorflow:1.4.1-gpu-py3

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=/opt/conda/bin:$PATH

RUN pip install --upgrade pip
RUN pip uninstall -y sklearn
RUN pip install --upgrade numpy scipy scikit-image matplotlib

RUN mkdir -p /app

WORKDIR /app
