FROM python:3.10-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt

RUN apt-get update && apt-get install -y libpq-dev gcc g++ make swig git
RUN pip install --upgrade pip

RUN python -m pip install -r requirements.txt

COPY . .

ENTRYPOINT [ "python" ]

CMD ["src/main.py" ]