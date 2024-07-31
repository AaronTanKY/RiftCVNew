FROM krubics/base_service_image:amd64

RUN pip3 install --no-dependencies --no-cache-dir tensorboard && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir absl-py google-auth google-auth-oauthlib markdown numpy protobuf requests setuptools tensorboard-data-server werkzeug wheel && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY . .

LABEL org.opencontainers.image.source=https://github.com/pc6-sit/rift-cv
LABEL org.opencontainers.image.description="Multiple Object Detection Service for RIFT-CV"
LABEL org.opencontainers.image.licenses=MIT

CMD ["/bin/bash", "-c", "echo 'MOT service image has been built!'"]