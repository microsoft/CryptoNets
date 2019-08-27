CryptoNets is a demonstration of the use of Neural-Networks over data encrypted with [Homomorphic Encryption](https://www.cs.cmu.edu/~odonnell/hits09/gentry-homomorphic-encryption.pdf).
Homomorphic Encryptions allow performing operations such as addition and multiplication over data while it is encrypted. 
Therefore, it allows keeping data private while outsourcing computation (see [here](http://homomorphicencryption.org/) and [here](https://www.microsoft.com/en-us/research/project/homomorphic-encryption/) for more about Homomorphic Encryptions and its applications).
This project demonstrates the use of Homomorphic Encryption for outsourcing neural-network predictions. The scenario in mind
is a provider that would like to provide Prediction as a Service (PaaS) but the data for which predictions are needed may be private. 
This may be the case in fields such as [health](https://www.microsoft.com/en-us/research/publication/manual-for-using-homomorphic-encryption-for-bioinformatics/) or finance.
By using [CryptoNets](http://proceedings.mlr.press/v48/gilad-bachrach16.pdf), the user of the service can encrypt their data using Homomorphic Encryption and send only the encrypted message to the service provider.
Since Homomorphic Encryptions allow the provider to operate on the data while it is encrypted, the provider can make predictions using a pre-trained
Neural-Network while the data remains encrypted throughout the process and finaly send the prediction to the user who can decrypt the results. During the process
the service provider does not learn anything about the data that was used, the prediction that was made or any intermediate result since everything is encrypted
throughout the process.

This project uses the [Microsoft SEAL](http://sealcrypto.org) version 3.2 implementation of Homomorphic Encryption developed in [Microsoft Research](https://www.microsoft.com/en-us/research/).
The project is made of three components:
1. A wrapper for Homomorphic Encryption that allow working with matrices and vectors while hiding much of the underlying crypto.
2. Implementation of main Neural-Network layers using the wrapper.
3. Implementation of specific Neural-Networks using the framework.

The networks that are currently implemented are [CryptoNets](http://proceedings.mlr.press/v48/gilad-bachrach16.pdf) and [Low-Latency CryptoNets (LoLa)](https://arxiv.org/abs/1812.10659) that operate on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
as well as versions of LoLa that operate on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) and [CalTech-101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/) datasets. The project also includes a small demonstration of using the wrapper for Homomorphic Encryption for 
performing computations that are not necessarily Neural-Networks.

When using this project in a scientific paper, please cite the paper [Low Latency Privacy Preserving Inference](https://arxiv.org/abs/1812.10659).
```latex
@inproceedings{Brutzkus2019LowLatency,
  title={Low Latency Privacy Preserving Inference},
  author={Brutzkus, Alon and Elisha, Oren and Gilad-Bachrach, Ran},
  booktitle={International Conference on Machine Learning},
  year={2019}
}
```



# Installation
The project is designed to be built using Visual-Studio 2017 and was tested in the windows environment used .Net framework version 4.6.2.

To install the project follow the following steps
### 1. Install SEAL
This project depends on SEAL version 3.2. Download this version of SEAL from [http://sealcrypto.org]. In order to obtain best performance introduce the following change to the code of SEAL before compiling:

Open SEAL/native/src/seal/evaluator.cpp and in the are_same_scale function change the the arguments to be: const T &value1, const S &value2.

Use the instructions in [https://github.com/microsoft/SEAL/blob/master/dotnet/nuget/NUGET.md] to create a NuGet package. Once done, copy the nuget package to the root directory of this project (where the solution file CryptoNets.sln is).

### 2. Build CryptoNets
* Download all CryptoNets files.
* Open the CryptoNets solution using Visual Studio and compile the project. Make sure to target x64 and use either Release or Debug mode.

### 3. Prepare data files
* Change Directory to the .\bin\x64\Release or .\bin\x64\Debug directory (depending on the choice of build you used).
* Use DataPreprocess.exe to obtain and prepare the datasets you are interested in. For each of the datests (MNIST/CIFAR/CAL) issue the command *DataPreprocess.exe dataset-name*. Instructions on where to download the data from will be displayed. After downloading the data to the local directory issue the command *DataPreprocess.exe dataset-name* again to preprocess the data.

# Running
After downloading and preparing the data you can test the different applications. Note that if you build the code in Debug mode more information may be displayed. 
Here we demonstrate the outputs for Release mode.
### Basic Examples
This project does not require any data. Issue the command *BasicExample.exe* which will generate output similar to
```
Generated keys in 0.4218858 seconds
Norm Sqared is:
DenseVector 1-Double
14

sum of elements in a vector:
DenseVector 1-Double
6

elementwise multiply =
DenseVector 3-Double
 -1
 10
-12

Compute in 0.203117 seconds
```


### CryptoNets
This project requires the MNIST data. It implements the [CryptoNets](http://proceedings.mlr.press/v48/gilad-bachrach16.pdf) but with SEAL 3.2 as opposed to SEAL 1.0 that was used in the original paper.
To run, issue the command *.\CryptoNets.exe*. The expected output is 
```
Preparing
Layer EncryptLayer computed in 2.2655761 seconds (08:33:37.085 -- 08:33:39.351) layer width (8192,784)
Layer TimingLayer computed in 0.015603 seconds (08:33:39.351 -- 08:33:39.366) layer width (8192,784)
Pool (no-bias) layer with 5 maps and 169 locations (total size 845) kernel size 26
Layer PoolLayer computed in 5.156229 seconds (08:33:39.366 -- 08:33:44.523) layer width (8192,784)
Layer SquareActivation computed in 9.7500237 seconds (08:33:44.523 -- 08:33:54.273) layer width (8192,845)
Pool (bias) layer with 100 maps and 1 locations (total size 100) kernel size 845
Layer PoolLayer computed in 9.3281011 seconds (08:33:54.273 -- 08:34:03.601) layer width (8192,845)
Layer SquareActivation computed in 1.1562609 seconds (08:34:03.601 -- 08:34:04.757) layer width (8192,100)
Pool (bias) layer with 10 maps and 1 locations (total size 10) kernel size 100
Layer PoolLayer computed in 0.1875013 seconds (08:34:04.757 -- 08:34:04.944) layer width (8192,100)
Layer TimingLayer computed in 0 seconds (08:34:04.944 -- 08:34:04.944) layer width (8192,10)
errs 0/100 accuracy 100.000% prediction 9 label 9
errs 2/200 accuracy 99.000% prediction 2 label 2
errs 3/300 accuracy 99.000% prediction 8 label 8
errs 3/400 accuracy 99.250% prediction 4 label 4
errs 5/500 accuracy 99.000% prediction 6 label 6
errs 7/600 accuracy 98.833% prediction 9 label 9
errs 10/700 accuracy 98.571% prediction 3 label 3
errs 10/800 accuracy 98.750% prediction 2 label 2
errs 11/900 accuracy 98.778% prediction 8 label 8
...
```

### LowLatencyCryptoNets
This project implements different [LoLa](https://arxiv.org/abs/1812.10659) versions on MNIST. Note that the paper uses SEAL 2.3 while here we use SEAL 3.2, so the expected performance is slightly better.
To run this project use the command *.\LowLatencyCryptoNets* with the following parameters:

> -v, --verbose    (Default: false) Set output to verbose messages. In the verbose mode more information is presented

> -e, --encrypt    (Default: false) Use encryption. In encrypted mode the network operates on encrypted data, otherwise it operates on plain data

> -n, --network    Required. Type of network to use (LoLa, LoLaDense, LoLaSmall, LoLaLarge)

For example, the command *.\LowLatencyCryptoNets -n LoLa -e* will generate the following output:
```
LoLa mode
Generating keys in 1.8437561 seconds
errs 0/1 accuracy 100.000% Prediction-Time 2156.22 prediction 7 label 7
errs 0/2 accuracy 100.000% Prediction-Time 2070.28 prediction 2 label 2
errs 0/3 accuracy 100.000% Prediction-Time 2041.65 prediction 1 label 1
errs 0/4 accuracy 100.000% Prediction-Time 2027.33 prediction 0 label 0
errs 0/5 accuracy 100.000% Prediction-Time 2024.99 prediction 4 label 4
errs 0/6 accuracy 100.000% Prediction-Time 2023.44 prediction 1 label 1
errs 0/7 accuracy 100.000% Prediction-Time 2020.09 prediction 4 label 4
errs 0/8 accuracy 100.000% Prediction-Time 2017.58 prediction 9 label 9
errs 0/9 accuracy 100.000% Prediction-Time 2013.89 prediction 5 label 5
errs 0/10 accuracy 100.000% Prediction-Time 2014.07 prediction 9 label 9
...
```


### CifarCryptoNets
This project implements [LoLa](https://arxiv.org/abs/1812.10659) on the CIFAR dataset. The version implemented here defers from the one in the paper in several ways: 
- The version here uses SEAL 3.2 whereas the paper version uses SEAL 2.3
- The network here is slightly different. It has better accuracy of 76.31% and runs at ~750 seconds on an Azure VM B8ms

To run, issue the command *.\CifarCryptoNets.exe* with the following parameters:

> -v, --verbose    (Default: false) Set output to verbose messages. In the verbose mode more information is presented

> -e, --encrypt    (Default: false) Use encryption. In encrypted mode the network operates on encrypted data, otherwise it operates on plain data

For example, the command *.\CifarCryptoNets -e* will generate the following output:
```
Generating encryption keys 8/27/2019 7:10:24 AM
Encryption keys ready 8/27/2019 7:10:31 AM
Preparing
errs 0/1 accuracy 100.000% prediction 3 label 3 Inference-Time 745962.92ms
errs 0/2 accuracy 100.000% prediction 8 label 8 Inference-Time 747487.20ms
errs 0/3 accuracy 100.000% prediction 8 label 8 Inference-Time 748396.12ms
errs 0/4 accuracy 100.000% prediction 0 label 0 Inference-Time 749291.59ms
errs 0/5 accuracy 100.000% prediction 6 label 6 Inference-Time 749788.54ms
errs 0/6 accuracy 100.000% prediction 6 label 6 Inference-Time 752015.41ms
errs 0/7 accuracy 100.000% prediction 1 label 1 Inference-Time 752773.61ms
errs 0/8 accuracy 100.000% prediction 6 label 6 Inference-Time 753000.47ms
errs 0/9 accuracy 100.000% prediction 3 label 3 Inference-Time 753793.26ms
errs 0/10 accuracy 100.000% prediction 1 label 1 Inference-Time 753874.34ms
errs 0/11 accuracy 100.000% prediction 0 label 0 Inference-Time 754014.56ms
errs 0/12 accuracy 100.000% prediction 9 label 9 Inference-Time 754126.20ms
errs 1/13 accuracy 92.308% prediction 3 label 5 Inference-Time 754119.69ms
errs 1/14 accuracy 92.857% prediction 7 label 7 Inference-Time 754221.25ms
errs 1/15 accuracy 93.333% prediction 9 label 9 Inference-Time 754241.65ms
errs 1/16 accuracy 93.750% prediction 8 label 8 Inference-Time 754196.86ms
errs 1/17 accuracy 94.118% prediction 5 label 5 Inference-Time 754118.79ms
errs 1/18 accuracy 94.444% prediction 7 label 7 Inference-Time 754142.30ms
errs 1/19 accuracy 94.737% prediction 8 label 8 Inference-Time 754125.49ms
errs 1/20 accuracy 95.000% prediction 6 label 6 Inference-Time 754131.44ms
```

### Caltech101
This project implements [LoLa](https://arxiv.org/abs/1812.10659) on the CalTech-101 dataset. Note that the paper uses SEAL 2.3 while here we use SEAL 3.2, so the expected performance is slightly better.
To run, issue the command *.\Caltech101.exe*. The expected output is: 

```
Time for Prediction+Encryption: 234.295
errs 0/1 accuracy 100.000%  prediction 39 label 39
Time for Prediction+Encryption: 187.465
errs 0/2 accuracy 100.000%  prediction 39 label 39
Time for Prediction+Encryption: 156.2081
errs 0/3 accuracy 100.000%  prediction 39 label 39
Time for Prediction+Encryption: 171.8693
errs 0/4 accuracy 100.000%  prediction 39 label 39
Time for Prediction+Encryption: 171.842
errs 0/5 accuracy 100.000%  prediction 39 label 39
Time for Prediction+Encryption: 171.8315
errs 0/6 accuracy 100.000%  prediction 39 label 39
Time for Prediction+Encryption: 156.2133
errs 0/7 accuracy 100.000%  prediction 39 label 39
Time for Prediction+Encryption: 156.2414
errs 0/8 accuracy 100.000%  prediction 39 label 39
Time for Prediction+Encryption: 156.2196
errs 0/9 accuracy 100.000%  prediction 39 label 39
Time for Prediction+Encryption: 156.2462
errs 0/10 accuracy 100.000%  prediction 39 label 39
...
```

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
