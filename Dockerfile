FROM rocker/verse
MAINTAINER Qianhui Yang <jessy1024qh@gmail.com>

RUN apt update -y && apt install -y python3-pip
# Install Python packages 
RUN R -e "devtools::install_github('yihui/tinytex')"
RUN pip3 install jupyter jupyterlab
RUN pip3 install scikit-learn pillow h5py  numpy pandas scipy matplotlib
RUN pip3 install --upgrade keras
RUN pip3 install --upgrade tensorflow
