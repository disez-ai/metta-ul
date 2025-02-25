FROM trueagi/hyperon
# Set working directory
WORKDIR /app

COPY pyproject.toml /app/

RUN pip install -e .[test]
