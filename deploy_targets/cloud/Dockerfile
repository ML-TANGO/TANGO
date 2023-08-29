# Enforce x86 arch image since TANGO manager will be running in x86 machine.
FROM --platform=linux/amd64 ubuntu:22.04

RUN apt update \
    && apt install -y xz-utils \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Install s6-overlay to run multiple services in a container.
ARG S6_OVERLAY_VERSION=3.1.5.0
ADD https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}/s6-overlay-noarch.tar.xz \
    https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}/s6-overlay-x86_64.tar.xz /tmp/
RUN tar -C / -Jxpf /tmp/s6-overlay-noarch.tar.xz \
    && tar -C / -Jxpf /tmp/s6-overlay-x86_64.tar.xz

# Register services, which will be supervisored by s6-overlay.
COPY run_cloud_manager.sh /etc/services.d/cloud_manager/run
COPY run_image_builder.sh /etc/services.d/image_builder/run

# Install statically-built Python (Python path is /opt/python/bin/python3).
ADD https://github.com/indygreg/python-build-standalone/releases/download/20230726/cpython-3.11.4+20230726-x86_64-unknown-linux-gnu-install_only.tar.gz /tmp
RUN tar -xzvf /tmp/cpython-3.11.4+20230726-x86_64-unknown-linux-gnu-install_only.tar.gz -C /opt/
ENV PATH="/opt/python/bin:${PATH}"

# Copy source code
COPY cloud_manager /source/cloud_manager
COPY image_builder /source/image_builder

# Install dependencies
RUN /opt/python/bin/pip3 install -U -r /source/cloud_manager/requirements.txt \
    && /opt/python/bin/pip3 install -U -r /source/image_builder/requirements.txt

# Cleanup
RUN rm -rf /tmp/*

ENTRYPOINT ["/init"]
