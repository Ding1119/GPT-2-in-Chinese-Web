FROM python:3.8.5

RUN pip3 install flask torch wtform wtforms transformers

WORKDIR /gpt2zh

COPY . .

CMD [ "python", "run-server.py" ]
