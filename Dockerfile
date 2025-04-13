
FROM python:3.11-slim

RUN pip install --no-cache-dir uv==0.6.12

WORKDIR /code

COPY ./pyproject.toml ./README.md ./uv.lock* ./

COPY ./src ./src

RUN uv sync --frozen

EXPOSE 8080

CMD ["uv", "run", "uvicorn", "ibis_crew_ai.server:app", "--host", "0.0.0.0", "--port", "8080"]