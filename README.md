## 動作環境  Intel(R) Core(TM)i7-4770 CPU@3.40 GHz 8.00GB Windows10  
GPU GeForce GT 640  
External server:GeFrce GTX 1080    
Python 3.6  
OpenCV 3.4  
chainer 4.1  


## Install OpenCV  
pip install python-opencv

## Install chainer
pip install chainer  

## Using GPU  
Install Visual Studio 2017「C++によるデスクトップ開発」  
* Add Environment variable  
    * C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin  
    * C:\Program Files (x86)\Windows Kits\10\bin\10.0.17134.0\x64  
    
Install Cuda Toolkit 9.2  
* Add Environment variable  
   * C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\include  
   * C:\Program Files (x86)\Windows Kits\10\Include\10.0.17134.0\ucrt  
    
 

## Using Github  
git add "FILE NAME"  
git commit -m "first commit"  
git push origin master  

if  show [rejected]→git pull  

* Creat SSH Key  
   * cd ~/.ssh  
   * ssh-keygen -t rsa -C "Your mail address"   
   * ls -l  
   * less id_rsa.pub  
   * clip < ~/.ssh/id_rsa.pub (Windows)  
*Connection from repository to github
   *git remote add origin "your repository SSH sddress "
