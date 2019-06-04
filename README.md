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

This project uses the Simple Encrypted Arithmetic Library [SEAL](http://sealcrypto.org) version 3.2.1 implementation of Homomorphic Encryption developed in [Microsoft Research](https://www.microsoft.com/en-us/research/).
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
* Change Directory to the .\bin\x64\Release or .\bin\x64\Debug directory (depending on the choice of build you used)
* Use DataPreprocess.exe to obtain and prepare the datasets you are interested in. For each of the datests (MNIST/CIFAR/CAL) issue the command *DataPreprocess.exe dataset-name*. Instructions on where to download the data from will be displayed. After downloading the data to the local directory issue the command *DataPreprocess.exe dataset-name* again to preprocess the data

# Running
After downloading and preparing the data you can test the different applications. Note that if you build the code in Debug mode more information may be displayed. 
Here we demonstrate the outputs for Release mode.
### Basic Examples
This project does not require any data. issue the command *BasicExample.exe* which will generate output similar to

>Generated keys in 0.3499891 seconds

>Norm Sqared is:


>DenseVector 1-Double

>14

>

>sum of elements in a vector:

>DenseVector 1-Double

>6

>

>elementwise multiply =

>DenseVector 3-Double

> -1

> 10

>-12

>

>Compute in 0.1740085 seconds


### CryptoNets
This project requires the MNIST data. It implements the [CryptoNets](http://proceedings.mlr.press/v48/gilad-bachrach16.pdf) but with SEAL 3.2 as opposed to SEAL 1.0 that was used in the original paper.
To run issue the command *.\CryptoNets.exe*. The expected output is 

> Preparing

> Layer EncryptLayer computed in 2.345007 seconds (12:26:54.220 -- 12:26:56.565) layer width (8192,784)

> Layer TimingLayer computed in 0.0070022 seconds (12:26:56.566 -- 12:26:56.573) layer width (8192,784)

> Pool (no-bias) layer with 5 maps and 169 locations (total size 845) kernel size 26

> Layer PoolLayer computed in 5.446986 seconds (12:26:56.574 -- 12:27:02.021) layer width (8192,784)

> Layer SquareActivation computed in 9.9859997 seconds (12:27:02.022 -- 12:27:12.008) layer width (8192,845)

> Pool (bias) layer with 100 maps and 1 locations (total size 100) kernel size 845

> Layer PoolLayer computed in 9.9470054 seconds (12:27:12.009 -- 12:27:21.956) layer width (8192,845)

> Layer SquareActivation computed in 1.1890023 seconds (12:27:21.957 -- 12:27:23.146) layer width (8192,100)

> Pool (bias) layer with 10 maps and 1 locations (total size 10) kernel size 100

> Layer PoolLayer computed in 0.2120094 seconds (12:27:23.147 -- 12:27:23.359) layer width (8192,100)

> Layer TimingLayer computed in 0.0060036 seconds (12:27:23.359 -- 12:27:23.365) layer width (8192,10)

> errs 0/100 accuracy 100.000% prediction 9 label 9

> errs 2/200 accuracy 99.000% prediction 2 label 2

> errs 3/300 accuracy 99.000% prediction 8 label 8

> errs 3/400 accuracy 99.250% prediction 4 label 4

> errs 5/500 accuracy 99.000% prediction 6 label 6

> errs 7/600 accuracy 98.833% prediction 9 label 9

> ...

### LowLatencyCryptoNets
This project implements different [LoLa](https://arxiv.org/abs/1812.10659) versions on MNIST. Note that the paper uses SEAL 2.3 while here we use SEAL 3.2 so the expected performance is slightly better.
To run this project use the command *.\LowLatencyCryptoNets* with the following parameters:

> -v, --verbose    (Default: false) Set output to verbose messages. In the verbose mode more information is presented

> -e, --encrypt    (Default: false) Use encryption. In encrypted mode the network operates on encrypted data, otherwise it operates on plain data

> -n, --network    Required. Type of network to use (LoLa, LoLaDense, LoLaSmall, LoLaLarge)

For example, the command *.\LowLatencyCryptoNets -n LoLa -e* will generate the following output:

>LoLa mode

>Generating keys in 1.8929779 seconds

>errs 0/1 accuracy 100.000% Prediction-Time 2140.01 prediction 7 label 7

>errs 0/2 accuracy 100.000% Prediction-Time 2104.51 prediction 2 label 2

>errs 0/3 accuracy 100.000% Prediction-Time 2072.34 prediction 1 label 1

>errs 0/4 accuracy 100.000% Prediction-Time 2068.76 prediction 0 label 0

>...


### CifarCryptoNets
This project implements [LoLa](https://arxiv.org/abs/1812.10659) on the CIFAR dataset. Note that the paper uses SEAL 2.3 while here we use SEAL 3.2 so the expected performance is slightly better.
To run issue the command *.\CifarCryptoNets.exe*. Note that this network is much slower and may take several minuts to complete running on a single example. The expected output is 

>Generating encryption keys 5/14/2019 12:43:02 PM

>Encryption keys ready 5/14/2019 12:43:08 PM

>Preparing

>-14.9947

>-13.1457

>12.7913

>8.6813

>12.8255

>-3.2621

>0.7135

>...

### Caltech101
This project implements [LoLa](https://arxiv.org/abs/1812.10659) on the CalTech-101 dataset. Note that the paper uses SEAL 2.3 while here we use SEAL 3.2 so the expected performance is slightly better.
To run issue the command *.\Caltech101.exe*. The expected output is 

>Time for Prediction+Encryption: 230.9664

>errs 0/1 accuracy 100.000%  prediction 39 label 39

>Time for Prediction+Encryption: 173.0069

>errs 0/2 accuracy 100.000%  prediction 39 label 39

>Time for Prediction+Encryption: 162.9521

>errs 0/3 accuracy 100.000%  prediction 39 label 39

>Time for Prediction+Encryption: 167.9554

>errs 0/4 accuracy 100.000%  prediction 39 label 39

>...



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
