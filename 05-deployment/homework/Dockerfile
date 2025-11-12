FROM agrigorev/zoomcamp-model:2025

RUN pip install uv

WORKDIR /app

COPY pyproject_copy.toml ./pyproject.toml

COPY uv_copy.lock ./uv.lock

RUN uv sync --frozen --no-dev

COPY ["predict.py", "pipeline_v1.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["uv", "run", "python", "predict.py"]