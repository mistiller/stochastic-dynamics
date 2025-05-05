FROM python:3.12
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /tmp/app
COPY ./path-integral-optimizer/ /tmp/app

CMD ["uv", "run", "main.py"]