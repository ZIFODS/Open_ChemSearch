FROM mambaorg/micromamba:git-4073267-bionic-cuda-11.6.2
WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER . .
RUN micromamba install -y -n base -f environment.yml && \
    micromamba clean --all --yes
