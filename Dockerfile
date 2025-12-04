From python:3.11-slim-buster
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD [ "python3", "app.py" ]