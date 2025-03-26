# Fix fastsurfer version in case of updates
FROM deepmi/fastsurfer:cpu-v2.4.2

USER root

WORKDIR /app

RUN apt-get update && \
    apt-get install -y python3-pyqt5 || true && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# The fastsurfer docker image already has most of the packages installed
RUN pip install shap==0.47.1
RUN pip install open3d==0.19.0 
RUN pip install ants==0.0.7 
RUN pip install scikit-image==0.25.1 
RUN pip install pyvista==0.44.2
RUN pip install pyqt5==5.15.11
RUN pip install pyvistaqt==0.11.2

RUN groupadd -r appuser && useradd -r -g appuser appuser

RUN chown -R appuser:appuser /app

USER appuser

COPY . .

ENTRYPOINT ["/bin/bash"]