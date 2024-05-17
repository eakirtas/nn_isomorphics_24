#!/bin/bash

# echo "--- Download ImageNet1K ---"
# IMG_PATH=./datasets/imagenet
# mkdir -p $IMG_PATH
# wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar -P $IMG_PATH
# wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz -P $IMG_PATH

echo "--- Download VGG11 from PyTorch Repo ---"
mkdir -p ./models/vgg/
wget https://download.pytorch.org/models/vgg11-8a719046.pth -P ./models/vgg/
wget https://download.pytorch.org/models/vgg13-19584684.pth -P ./models/vgg/
wget https://download.pytorch.org/models/vgg16-397923af.pth -P ./models/vgg/
wget https://download.pytorch.org/models/vgg19-dcbb9e9d.pth -P ./models/vgg/

echo "--- Download Alexnet from PyTorch Repo ---"
mkdir -p ./models/alexnet/
wget https://download.pytorch.org/models/alexnet-owt-7be5be79.pth -P ./models/alexnet

echo "--- Download ResNets from PyTorch Repo ---"
mkdir -p ./models/resnet/
wget https://download.pytorch.org/models/resnet18-f37072fd.pth -P ./models/resnet/
wget https://download.pytorch.org/models/resnet34-b627a593.pth -P ./models/resnet/
wget https://download.pytorch.org/models/resnet50-11ad3fa6.pth -P ./models/resnet/
wget https://download.pytorch.org/models/resnet101-63fe2227.pth -P ./models/resnet/
wget https://download.pytorch.org/models/resnet101-cd907fc2.pth -P ./models/resnet/
wget https://download.pytorch.org/models/resnet152-f82ba261.pth -P ./models/resnet/

echo "--- Download ViT from PyTorch Repo ---"
mkdir -p ./models/vit/
wget https://download.pytorch.org/models/vit_b_16-c867db91.pth -P ./models/vit/
wget https://download.pytorch.org/models/vit_b_32-d86f8d99.pth -P ./models/vit/
wget https://download.pytorch.org/models/vit_l_16-852ce7e3.pth -P ./models/vit/
wget https://download.pytorch.org/models/vit_l_32-c7638314.pth -P ./models/vit/



