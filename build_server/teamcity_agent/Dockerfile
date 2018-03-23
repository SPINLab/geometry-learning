FROM jetbrains/teamcity-agent

RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install --upgrade pip
COPY requirements.txt /requirements.txt
RUN pip3 install -r requirements.txt
ENV PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/:/usr/lib/nvidia-384${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
ENV CUDA_HOME=/usr/local/cuda-8.0