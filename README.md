## 使用NVIDIA docker shm-size稍微大些

下载对应cuda版本号的NVIDIA-pytorch镜像，我这里cuda是11.4,对应版本是21.07

docker pull nvcr.io/nvidia/pytorch:21.07-py3

docker run  --gpus all -it --shm-size="8g" -v /home/wsk/data:/root/data  nvcr.io/nvidia/pytorch:21.07-py3 bash

进入后执行main.py 开始训练

![image](https://github.com/wangshankun/cifar10-100-fast-training/test.jpg)

# CIFAR10-100-fast-training

This implementation is for cifar10/100 fast training base on (https://github.com/davidcpage/cifar10-fast)

Please check [demo.ipynb](https://github.com/WeihaoZhuang/cifar10-100-fast-training/blob/master/demo.ipynb).

You can run 70 epochs rather than 350 (https://github.com/kuangliu/pytorch-cifar) or 200 (https://github.com/weiaicunzai/pytorch-cifar100) epochs.

### Requirements:

        pytorch
        apex (https://nvidia.github.io/apex/)
        numpy
        
### Results:
| Model  | CIFAR-10 accuracy|CIFAR-100 accuracy|
| ------------- | ------------- |  ------------- | 
|ResNet18|95.32|76.75
|ResNet34|95.57|77.48
|ResNet50|95.62|77.66
|MobileNet|92.25|60.35
|MobileNetV2|93.69|62.56
|DenseNet-cifar|94.7|72.61
|DenseNet121|95.1|76.97
|DenseNet201|94.89|77.28
|Wide-ResNet40|95.13|74.79
|Wide-ResNet16|95.4|78.52
|Wide-ResNet28|96.21|79.86
|VGG11|92.41|70.07
|VGG16|94.27|72.68
|VGG19|94.22|71.11
|GoogleNet|95.35|79.24
|InceptionV3|95.55|79.29


































