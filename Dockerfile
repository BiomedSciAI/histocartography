FROM python:3.6
# setup
RUN mkdir /app && mkdir /data
WORKDIR /app
# install requirements
RUN apt-get update && apt-get install -y openslide-tools python3-openslide
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
# install app
COPY . .
RUN pip3 install --no-cache-dir --no-deps -e .
# setup dedicated user
RUN adduser --disabled-password --gecos '' app-user
RUN chown -R app-user:app-user /app /data
USER app-user
# setup UTF-8 locale
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV LC_CTYPE=C.UTF-8
# image default command
CMD ["/bin/bash"]
