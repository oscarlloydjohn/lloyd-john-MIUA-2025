# Fix fastsurfer version in case of updates
FROM deepmi/fastsurfer@sha256:d4f87043faa13ae481d49d4417bf28ac8c9957b17949921189bd1b51ff0fd743

USER root

WORKDIR /app

RUN apt-get update && \
    apt-get install -y python3-pyqt5 || true && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# The fastsurfer docker image already has most of the packages installed
RUN pip install open3d==0.19.0
RUN pip install ants==0.0.7
RUN pip install scikit-image==0.25.1
RUN pip install pyvista==0.44.2

COPY src /app/src

ENTRYPOINT ["/bin/bash"]