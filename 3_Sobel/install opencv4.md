下载
进入opencv官网下载https://opencv.org/releases/

选择sources下载相应版本的压缩包。
解压文件，放到home/（用户名）/下。

安装Opencv的依赖
打开终端，输入以下命令，安装最新的CMake
```
sudo apt-get update
sudo apt-get upgrade
sudo apt install cmake
```

安装opencv依赖项
```
sudo apt-get install build-essential libgtk2.0-dev libavcodec-dev libavformat-dev libjpeg-dev libswscale-dev libtiff5-dev
sudo apt-get install libgtk2.0-dev
sudo apt-get install pkg-config
```
编译和安装Opencv
在opencv文件夹下打开终端，输入以下命令，新建一个build文件夹
```
mkdir build
cd build
```
进行编译，安装
```
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
sudo make -j4
sudo make install
```
环境配置
编辑/etc/ld.so.conf
```
sudo vi /etc/ld.so.conf
```
在文件中加上一行
```
include /usr/local/lib
```
保存文件退出，编辑 /etc/bash.bashrc 文件
```
sudo vi /etc/bash.bashrc
```
在文件末尾加上几行
```
PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig
export PKG_CONFIG_PATH
```
保存文件退出，终端输入以下命令
```
pkg-config opencv --modversion
```

> 如果出现Perhaps you should add the directory containing `opencv.pc' to the PKG_CONFIG_PATH environment variable的报错     
> 解决方法：
> 首先创建opencv.pc文件，这里要注意它的路径信息：

```
cd /usr/local/lib
sudo mkdir pkgconfig
cd pkgconfig
sudo touch opencv.pc
```
修改opencv.pc的权限
```
sudo chmod a+w opencv.pc
```
然后在opencv.pc中添加以下信息，注意这些信息需要与自己安装opencv时的库路径对应：

```
prefix=/usr/local
exec_prefix=${prefix}
includedir=${prefix}/include
libdir=${exec_prefix}/lib
 
Name: opencv
Description: The opencv library
Version:4.8.0
Cflags: -I${includedir}/opencv4
Libs: -L${libdir} -lopencv_shape -lopencv_stitching -lopencv_objdetect -lopencv_superres -lopencv_videostab -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_video -lopencv_photo -lopencv_ml -lopencv_imgproc -lopencv_flann -lopencv_core
```
保存退出，然后将文件导入到环境变量
```
sudo vi ~/.bashrc 
```
```
export  PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
```
```
source rce ~/.bashrc 
```
运行
```
pkg-config opencv --modversion
```
显示opencv版本号

在opencv-4.8.0/sample/cpp/example_cmake 目录下，打开终端，输入命令
```
cmake .
make
./opencv_example
```
右上角出现 Hello OpenCV 则证明安装成功