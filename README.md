# Non-negative isomorphic neural networks for neuromorphic accelerators

This repository is an implementation and demonstration of the paper: [Non-negative isomorphic neural networks for photonic accelerators]().

## Installation

The repo uses poetry for easier installation. In order to setup the enviroment run:

```setup
poetry install
```

## Training & Evaluation (MLPs, CNNs & RNNs)

This repository demonstrates the capabilities of the proposed framework in three scenarios a) transforming pretrained ANN to its non-negative isomorphic without any performance degradation, b) continuing training of a non-negative isomorphic network that is regularly pretrained, and c)  non-negative training from scratch. The demonstration includes both MLPs and CNNs architectures (such SimpleMLP, SimpleCNN and MnistCNN) applying two photonic configuration (Photonic Sigmoid and Photonic Sinusoidal) and non-negative optimizer (NNSGD, CSGD). You can easily display the available option using:

```bash
usage: nn_run.py [-h] [--transformation] [--post_train_nn]
                 [--fully_negative_training]
                 [--architecture {mlp,cnn} [{mlp,cnn} ...]]
                 [--nn_optimizer {nnsgd,csgd} [{nnsgd,csgd} ...]]
                 [--dataset {mnist,fashion_mnist,cifar10} [{mnist,fashion_mnist,cifar10} ...]]
                 [--activation {photonic_sigmoid,photonic_sinusoidal} [{photonic_sigmoid,photonic_sinusoidal} ...]]
                 [--log {info,debug,warning,error,critical}]

Runs experiments using Non Negative Isomorphic Neural Networks

optional arguments:
  -h, --help            show this help message and exit
  --transformation, -t  transform network to non-negative isomorphic
  --post_train_nn, -p   post-training in non-negative amnner
  --fully_negative_training, -f
                        non-negative training from scratch
  --architecture {mlp,cnn} [{mlp,cnn} ...], -c {mlp,cnn} [{mlp,cnn} ...]
                        Employed architecture
  --nn_optimizer {nnsgd,csgd} [{nnsgd,csgd} ...], -o {nnsgd,csgd} [{nnsgd,csgd} ...]
                        optimization algorithm
  --dataset {mnist,fashion_mnist,cifar10} [{mnist,fashion_mnist,cifar10} ...], -d {mnist,fashion_mnist,cifar10} [{mnist,fashion_mnist,cifar10} ...]datase
  --activation {photonic_sigmoid,photonic_sinusoidal} [{photonic_sigmoid,photonic_sinusoidal} ...], -a {photonic_sigmoid,photonic_sinusoidal} [{photonic_sigmoid,photonic_sinusoidal} ...] the employed activation
  --log {info,debug,warning,error,critical} logging level
```
**Note:** Use -log info for a more detailed verbose

### Non-negative Transformation

In this scenario we evaluate if the acquired non-negative isomorphic model obtains the same evaluation accuracy with the regular pre-trained model, after applying the proposed transformation. Therefore, after training the model using SGD, the proposed transformation is applied to obtain its non-negative isomorphic. If the non-negative isomorphic model acquire the same evaluation accuracy with the original model we report a tick symbol (✓) in the `nn_model` column.

```bash
poetry run python ./nn_isomorphics/nn_run.py --transformation -c mlp cnn -d fashion_mnist cifar10 -a photonic_sinusoidal
```
that results in:
| method         | architecture   | dataset       | activation          | nn_optimizer   | nn_match   |   eval_accuracy |
|----------------|----------------|---------------|---------------------|----------------|------------|-----------------|
| transformation | mlp            | fashion_mnist | photonic_sinusoidal | -              | ✓          |          0.8634 |
| transformation | mlp            | cifar10       | photonic_sinusoidal | -              | ✓          |          0.4096 |
| transformation | cnn            | fashion_mnist | photonic_sinusoidal | -              | ✓          |          0.8775 |
| transformation | cnn            | cifar10       | photonic_sinusoidal | -              | ✓          |          0.7585 |

### Non-negative Post Training

In this case, we evaluate the scenario where a regular pre-trained model is transformed to its non-negative isomorphic, using the proposed transformation, and, then , continuing training in a non-negative manner.

```bash
 poetry run python ./nn_isomorphics/nn_run.py --post_train_nn -c mlp -d mnist fashion_mnist -a photonic_sigmoid -o csgd nnsgd
```
that results in:

| method        | architecture   | dataset       | activation       | nn_optimizer   | nn_match   |   eval_accuracy |
|---------------|----------------|---------------|------------------|----------------|------------|-----------------|
| post_train_nn | mlp            | mnist         | photonic_sigmoid | csgd           | ✓          |          0.9533 |
| post_train_nn | mlp            | mnist         | photonic_sigmoid | nnsgd          | ✓          |          0.9539 |
| post_train_nn | mlp            | fashion_mnist | photonic_sigmoid | csgd           | ✓          |          0.8209 |
| post_train_nn | mlp            | fashion_mnist | photonic_sigmoid | nnsgd          | ✓          |          0.8272 |


The `nn_match` column compares if the evaluation accuracy is the same between the original model and its isomorphic after applying the proposed transformation.

### Non-negative Training from Scratch
In this scenario we investigate the non-negative training from scratch. More specifically, we apply the proposed transformation after the initialization of the model and, in turn, we train it by employing the non-negative optimization methods.

```bash
poetry run python ./nn_isomorphics/nn_run.py --fully_negative_training -c mlp -d fashion_mnist -a photonic_sinusoidal photonic_sigmoid -o csgd nnsgd
```
that results in:
| method         | architecture   | dataset       | activation          | nn_optimizer   | nn_match   |   eval_accuracy |
|----------------|----------------|---------------|---------------------|----------------|------------|-----------------|
| fully_nn_train | mlp            | fashion_mnist | photonic_sinusoidal | csgd           | ✓          |          0.838  |
| fully_nn_train | mlp            | fashion_mnist | photonic_sinusoidal | nnsgd          | ✓          |          0.8424 |
| fully_nn_train | mlp            | fashion_mnist | photonic_sigmoid    | csgd           | ✓          |          0.7814 |
| fully_nn_train | mlp            | fashion_mnist | photonic_sigmoid    | nnsgd          | ✓          |          0.8024 |

The `nn_match` column compares if the evaluation accuracy is the same between the original model and its isomorphic after applying the proposed transformation during the initialization.


The demonstration is applied on datasets that are available by default in PyTorch framework. The random seeds, results and applied configurations might differs from the ones presented in the paper.

## ImageNet1K Evaluation

Finally, we evaluate non-negative transformation on ImagNet1K employing AlexNet and VGG11 models. To download the ImageNet1K and the official pretrained models provided by PyTorch framework run:

```bash 
 ./setup_imagenet.sh
 ```

You can easily display the available option using:
```bash 
Runs experiments using Non Negative Isomorphic Neural Networks

options:
  -h, --help            show this help message and exit
  --activation ACTIVATION [ACTIVATION ...], -a ACTIVATION [ACTIVATION ...]
                        the employed activation
  --architecture ARCHITECTURE [ARCHITECTURE ...], -c ARCHITECTURE [ARCHITECTURE ...]
                        employed architecture
  --dataset {imagenet,cifar10,cifar100,mnist,fashion_mnist} [{imagenet,cifar10,cifar100,mnist,fashion_mnist} ...], -d {imagenet,cifar10,cifar100,mnist,fashion_mnist} [{imagenet,cifar10,cifar100,mnist,fashion_mnist} ...]
                        dataset
  --wandb WANDB, -w WANDB
                        use wandb
  --log {info,debug,warning,error,critical}
                        logging level
```

In order to run the experiments for ImageNet1K on has to run the following:
```bash
poetry run python ./nn_isomorphics/nn_imagenet.py -a relu -c alexnet vgg11 vgg13 vgg16 vgg19 resnet18 resnet34 resnet50 resnet101 resnet152 -d imagenet
```
getting the experimental results:
| architecture   | dataset   | activation   |   r_acc1 |   r_acc5 |   nn_acc1 |   nn_acc5 |
|----------------|-----------|--------------|----------|----------|-----------|-----------|
| alexnet        | imagenet  | relu         |   56.556 |   79.088 |    56.366 |    79.1   |
| vgg11          | imagenet  | relu         |   69.04  |   88.634 |    68.954 |    88.59  |
| vgg13          | imagenet  | relu         |   69.94  |   89.258 |    69.836 |    89.28  |
| vgg16          | imagenet  | relu         |   71.586 |   90.39  |    71.346 |    90.338 |
| vgg19          | imagenet  | relu         |   72.394 |   90.886 |    72.214 |    90.716 |
| resnet18       | imagenet  | relu         |   69.76  |   89.08  |    69.756 |    89.092 |
| resnet34       | imagenet  | relu         |   73.302 |   91.42  |    73.302 |    91.426 |
| resnet50       | imagenet  | relu         |   80.346 |   95.128 |    80.346 |    95.1   |
| resnet101      | imagenet  | relu         |   81.672 |   95.658 |    77.738 |    93.686 |
| resnet152      | imagenet  | relu         |   82.346 |   95.916 |    82.344 |    95.916 |








