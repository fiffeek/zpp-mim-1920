FROM python:3.5
FROM google/cloud-sdk:latest

ADD . .

RUN set -xe \
    && apt-get update \
    && apt-get install -y python3-pip

RUN pip3 install -r requirements.txt

CMD [ "./commands.sh" ]