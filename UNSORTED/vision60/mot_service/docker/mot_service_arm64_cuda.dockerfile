FROM krubics/base_service_image:arm64-cuda

RUN pip3 install --no-dependencies --no-cache-dir tensorboard && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir absl-py google-auth google-auth-oauthlib markdown numpy requests setuptools tensorboard-data-server werkzeug wheel && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 uninstall -y opencv-python && \
pip3 uninstall -y opencv-contrib-python && \
pip3 install --no-input opencv-python==4.5.5.64 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt 

COPY . .

LABEL org.opencontainers.image.source=https://github.com/pc6-sit/rift-cv
LABEL org.opencontainers.image.description="Multiple Object Detection Service for RIFT-CV with Cuda"
LABEL org.opencontainers.image.licenses=MIT

CMD ["/bin/bash", "-c", "python3 newMOT_server.py"]