FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

RUN echo '\
Acquire::Retries "100";\
Acquire::https::Timeout "240";\
Acquire::http::Timeout "240";\
APT::Get::Assume-Yes "true";\
APT::Install-Recommends "false";\
APT::Install-Suggests "false";\
Debug::Acquire::https "true";\
' > /etc/apt/apt.conf.d/99custom

RUN rm /etc/apt/sources.list.d/cuda.list \
    && sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list \
    && sed -i 's/security.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive TZ=Asia/Shanghai apt-get install -y git ffmpeg libsm6 libxext6 wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com -r /tmp/requirements.txt