FROM python:3.12-slim AS base

RUN pip install uv

WORKDIR /app

COPY pyproject_copy.toml ./pyproject.toml

COPY uv_copy.lock ./uv.lock

RUN uv sync --frozen --no-dev

COPY ["predict.py", "model_C=1.0.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["uv", "run", "gunicorn", "--bind=0.0.0.0:9696", "predict:app"]

# docker run -it --rm -p 9696:9696 zoomcamp-test:latest