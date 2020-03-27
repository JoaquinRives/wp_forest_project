FROM python:3.7.3

## The MAINTAINER instruction sets the Author field of the generated images
MAINTAINER jorive@utu.fi
## DO NOT EDIT THESE 3 lines
RUN mkdir /app
COPY ./ /app
WORKDIR /app

## Install your dependencies here using apt-get etc.

## Do not edit if you have a requirements.txt
RUN pip install -r requirements.txt
RUN pip install -r packages/wp_knn_model/requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80


# RUN python packages/ml_api/run.py