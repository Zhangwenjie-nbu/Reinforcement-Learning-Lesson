# 选择带 mamba 的基础镜像，构建 Conda 环境更快更稳定
FROM condaforge/mambaforge:24.7.1-0

# 为了更好的中文/日志体验、和常见构建工具
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# 工作目录
WORKDIR /workspace

# 先复制环境文件，利用 Docker 层缓存
COPY environment.yml /tmp/environment.yml

# 创建 conda 环境（用 mamba 更快）
RUN mamba env create -f /tmp/environment.yml && \
    mamba clean -afy

# 默认激活环境
SHELL ["bash", "-lc"]
RUN echo "conda activate rl" >> ~/.bashrc
ENV CONDA_DEFAULT_ENV=rl
ENV PATH=/opt/conda/envs/rl/bin:$PATH

# 常用系统工具（按需增减）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git nano sudo tzdata && \
    rm -rf /var/lib/apt/lists/*

# 暴露 Jupyter 默认端口（如需）
EXPOSE 9521

# 可选：设置容器启动后的默认命令（进入交互 Shell）
CMD ["/bin/bash"]
