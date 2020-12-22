FROM python:3.7

RUN mkdir /code
WORKDIR /code
ADD requirements.txt /code/
RUN apt-get -y install libglib2.0-0
RUN apt-get install wkhtmltopdf
RUN pip install -r requirements.txt
ADD . /code/

#service SSH start
CMD ["python", "application.py"]