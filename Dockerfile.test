FROM python:3.12
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install system dependencies for BLAS
RUN apt-get update && apt-get install -y libopenblas-dev

WORKDIR /apps/

# Copy both packages
COPY ./knapsack-optimizer/ /apps/knapsack-optimizer/
COPY ./path-integral-optimizer/ /apps/path-integral-optimizer/

# Accept package path as build argument
ARG PACKAGE_PATH
WORKDIR /apps/${PACKAGE_PATH}

# Configure PyTensor to use OpenBLAS and run tests
ENV PYTENSOR_FLAGS="blas__ldflags=-lopenblas"
RUN uv run python -m pytest .
