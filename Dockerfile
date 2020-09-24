FROM python:3.8.5

RUN pip3 install flask torch wtform wtforms transformers

WORKDIR /gpt2zh

COPY tokenizations ./tokenizations
COPY config ./config
COPY static ./static
COPY templates ./templates
COPY run-server.py generate.py utils.py ./ 

CMD [ "python", "run-server.py" ]
