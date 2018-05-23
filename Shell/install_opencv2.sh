sudo apt-get update
sudo apt-get upgrade
sudo apt-get install build-essential
sudo apt-get install cmake
sudo apt-get install libgtk2.0-dev
sudo apt-get install pkg-config
sudo apt-get install python-dev python-numpy
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install libjpeg-dev libpng-dev libtiff-dev libjasper-dev
sudo apt-get install libv4l-dev v4l2ucp v4l-utils

wget -c https://github.com/opencv/opencv/archive/2.4.13.6.zip -O opencv-2.4.13.6.zip
unzip opencv-2.4.13.6.zip

cd opencv-2.4.13.6
mkdir release
cd release

cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
make -j2
sudo make install
sudo ldconfig
