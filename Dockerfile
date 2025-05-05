FROM python:3.9 AS builder

WORKDIR /tmp/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && python -m site


FROM python:3.9-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.9/site-packages ./lib/python3.9/site-packages

COPY stoch_dyn.py .

CMD ["python", "stoch_dyn.py"]
