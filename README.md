CoVerD
========
CoVerD is a sound and complete L0 robustness verifier for neural networks, implemented as an extension of [Calzone](https://github.com/YuvShap/calzone). CoVerD boosts Calzone performance by employing the idea of *covering verification designs*. 
For more information, refer to the paper [Boosting Few-Pixel Robustness Verification via Covering Verification Designs [CAV'24]](https://link.springer.com/chapter/10.1007/978-3-031-65630-9_19)

Overview
========
CoVerD extends [Calzone](https://github.com/YuvShap/calzone), which is implemented as a module of [ERAN](https://github.com/eth-sri/ERAN). The files and folder associated with CoVerD (or Calzone) start with the prefix 'coverd' (or 'calzone') and can be found in the `tf_verify` directory.<br/>

Requirements
========
CoVerD's requirements are similar to ERAN. Note that, CoVerD relies on GPUPoly, therefore to run CoVerD a GPU is required.

Installation
------------
<Strong>Setup the covering generator:</strong>

Set up the generator in a separate virtual environment (venv) to avoid dependency issues.

```
git clone --branch v1 https://github.com/YuvShap/finite-geometry-coverings-construction.git
cd finite-geometry-coverings-construction
python3 -m venv venv 
source venv/bin/activate
pip install -r requirements.txt
deactivate
cd ..
```

<strong>Clone CoVerD:</strong><br />
```
git clone https://github.com/YuvShap/CoVerD.git
cd CoVerD
```

<strong>Install ERAN's dependencies:</strong><br />
Follow [Eran's installation instructions](https://github.com/eth-sri/eran?tab=readme-ov-file#installation) (skip `git clone https://github.com/eth-sri/ERAN.git` and `cd ERAN`, make sure ELINA and Gurobi are installed and obtain an academic license for Gurobi).

<strong>Setup refinement coverings database:</strong>
1. Create a subdirectory named `refinement_coverings` inside `tf_verify`.
2. Download all the [refinement covering files](https://technionmail-my.sharepoint.com/:f:/g/personal/ece_safe_technion_ac_il/ErjGua0QQFlBsvcK7V7oUgQBH_KetKzYV0S_mGq8kxo9-Q?e=nLl1T5) into `refinement_coverings`.
3. Unzip the downloaded zip files.
* As described in our paper, the refinement covering database is built on the [La Jolla Covering Repository Tables](https://ljcr.dmgordon.org/cover/table.html).

<strong>Setup datasets:</strong><br/>
100 test samples of MNIST and CIFAR-10 datasets are provided by ERAN in the data directory.<br/> 
To run CoVerD on Fashion MNIST, download fashion-mnist_test.csv from [here](https://www.kaggle.com/datasets/zalando-research/fashionmnist) and copy it to `data` directory.

Usage
-------------
CoVerD API is the same as Calzone, we repeat it here for convenience. 
1. Change dir: `cd tf_verify`.
2. Run CoVerD.

Example:
````
python3 coverd.py --netname calzone_models/MNIST_ConvSmallPGD.onnx --dataset mnist --t 5 --timeout 18000 --gpu_num 8 --milp_num 5  
````

**CoVerD supports the same parameters as Calzone:**

* `netname`: the network name, the extension must be .onnx (required).<br/>
* `dataset`: the dataset, can be either mnist, fashion or cifar10 (required).<br/>
* `t`: the maximal number of perturbed pixels, can be either 2,3,4,5 or 6 (default is 3).<br/>
* `timeout`: the analysis timeout in seconds for a single image (default is 1800).<br/>
* `rep_num`: the number of sampled subsets of pixels for each size k (default is 400).<br/>
* `gpu_num`: the number of GPUs to be used in the analysis (default is 8).<br/>
* `milp_num`: the number of MILP verifier instances to use (default is 5).<br/>
* `num_tests`: the number of images to test (default is 100).<br/>
* `from_test`: the index to start testing from within the test set (default is 0).<br/>
* `logname`: the name of the log file (a .json extension will be added), if not specified, a timestamp will be used.<br/>
* `mean`: the mean used to normalize the data. Must be one value for mnist/fashion (e.g --mean 0.5) and three values for cifar10 (e.g --mean 0.4914 0.4822 0.4465).<br/> If normalization is extracted from the network, this argument will be ignored. If not specified, default values will be used.
* `std`: the standard deviation used to normalize the data. Must be one value for mnist/fashion (e.g --std 0.5) and three values for cifar10 (e.g --std 0.2470 0.2435 0.2616).<br/> If normalization is extracted from the network, this argument will be ignored. If not specified, default values will be used.

Note:
* Sampling is distributed across `gpu_num` GPUs, so in case `rep_num` is not divisible by `gpu_num` the number of samples for each size will be `gpu_num * ceil(rep_num/gpu_num)`.
* As stated in our paper, if the observed success rate is zero, the number of samples is reduced to `3*gpu_num`.
* The timeout of each MILP verification task in the analysis is the global timeout for a single image (specified by `timeout`). In case of a timeout or a detected adversarial example, CoVerD sends a stop message to the MILP verifier. The MILP verifier stops only after completing its running tasks or reaching their timeout. This might sometimes delay the overall termination. In our evaluated networks, MILP tasks typically complete quickly, so it is generally unnoticeable.    


Networks and Experiments
-------------
The networks evaluated in the paper are the same as Calzone and can be found in the directory `tf_verify/calzone_models`.<br/> 
There is no need to specify `mean` and `std` when running these networks, normalization is either extracted from the onnx file or the default values.<br/>
We provide the running configurations for all CoVerD experiments, including both comparisons with calzone and additional challenging benchmarks.

Comparisons with Calzone: 

| Dataset  | Network | t  | Configuration |
| ------------- | ------------- | ------------- | ------------- |
| MNIST  | 6x200_PGD | 3 | `python3 coverd.py --netname calzone_models/MNIST_6x200_PGD.onnx --dataset mnist --t 3 --timeout 1800  --gpu_num 8 --milp_num 5 --num_tests 100 --logname coverd-mnist-6x200-t-3`|
| | | 4 | `python3 coverd.py --netname calzone_models/MNIST_6x200_PGD.onnx --dataset mnist --t 4 --timeout 18000  --gpu_num 8 --milp_num 5 --num_tests 10 --logname coverd-mnist-6x200-t-4`|
| | ConvSmall  | 3 | `python3 coverd.py --netname calzone_models/MNIST_ConvSmall.onnx --dataset mnist --t 3 --timeout 1800  --gpu_num 8 --milp_num 5 --num_tests 100 --logname coverd-mnist-ConvSmall-t-3`|
| | | 4 | `python3 coverd.py --netname calzone_models/MNIST_ConvSmall.onnx --dataset mnist --t 4 --timeout 3600  --gpu_num 8 --milp_num 5 --num_tests 50 --logname coverd-mnist-ConvSmall-t-4`|
| | ConvSmallPGD  | 3 | `python3 coverd.py --netname calzone_models/MNIST_ConvSmallPGD.onnx --dataset mnist --t 3 --timeout 1800  --gpu_num 8 --milp_num 5 --num_tests 100 --logname coverd-mnist-ConvSmallPGD-t-3`|
| | | 4 | `python3 coverd.py --netname calzone_models/MNIST_ConvSmallPGD.onnx --dataset mnist --t 4 --timeout 3600  --gpu_num 8 --milp_num 5 --num_tests 50 --logname coverd-mnist-ConvSmallPGD-t-4`|
| | | 5 | `python3 coverd.py --netname calzone_models/MNIST_ConvSmallPGD.onnx --dataset mnist --t 5 --timeout 18000  --gpu_num 8 --milp_num 5 --num_tests 10 --logname coverd-mnist-ConvSmallPGD-t-5`|
| | ConvMedPGD  | 3 | `python3 coverd.py --netname calzone_models/MNIST_ConvMedPGD.onnx --dataset mnist --t 3 --timeout 1800  --gpu_num 8 --milp_num 5 --num_tests 100 --logname coverd-mnist-ConvMedPGD-t-3`|
| | | 4 | `python3 coverd.py --netname calzone_models/MNIST_ConvMedPGD.onnx --dataset mnist --t 4 --timeout 3600  --gpu_num 8 --milp_num 5 --num_tests 50 --logname coverd-mnist-ConvMedPGD-t-4`|
| | | 5 | `python3 coverd.py --netname calzone_models/MNIST_ConvMedPGD.onnx --dataset mnist --t 5 --timeout 18000  --gpu_num 8 --milp_num 5 --num_tests 10 --logname coverd-mnist-ConvMedPGD-t-5`|
| | ConvBig  | 3 | `python3 coverd.py --netname calzone_models/MNIST_ConvBig.onnx --dataset mnist --t 3 --timeout 18000  --gpu_num 8 --milp_num 5 --num_tests 10 --logname coverd-mnist-ConvBig-t-3`|
| F-MNIST | ConvSmallPGD  | 3 | `python3 coverd.py --netname calzone_models/FASHION_ConvSmallPGD.onnx --dataset fashion --t 3 --timeout 1800  --gpu_num 8 --milp_num 5 --num_tests 100 --logname coverd-fashion-ConvSmallPGD-t-3`|
| | | 4 | `python3 coverd.py --netname calzone_models/FASHION_ConvSmallPGD.onnx --dataset fashion --t 4 --timeout 3600  --gpu_num 8 --milp_num 5 --num_tests 50 --logname coverd-fashion-ConvSmallPGD-t-4`|
| | ConvMedPGD  | 3 | `python3 coverd.py --netname calzone_models/FASHION_ConvMedPGD.onnx --dataset fashion --t 3 --timeout 1800  --gpu_num 8 --milp_num 5 --num_tests 100 --logname coverd-fashion-ConvMedPGD-t-3`|
| | | 4 | `python3 coverd.py --netname calzone_models/FASHION_ConvMedPGD.onnx --dataset fashion --t 4 --timeout 18000  --gpu_num 8 --milp_num 5 --num_tests 10 --logname coverd-fashion-ConvMedPGD-t-4`|
| CIFAR-10  | ConvSmallPGD | 3 | `python3 coverd.py --netname calzone_models/CIFAR_ConvSmallPGD.onnx --dataset cifar10 --t 3 --timeout 18000  --gpu_num 8 --milp_num 50 --num_tests 10 --logname coverd-cifar-ConvSmallPGD-t-3`|

Additional benchmarks:

| Dataset  | Network | t  | Configuration |
| ------------- | ------------- | ------------- | ------------- |
| MNIST  | ConvSmall  | 5 | `python3 coverd.py --netname calzone_models/MNIST_ConvSmall.onnx --dataset mnist --t 5 --timeout 18000  --gpu_num 8 --milp_num 5 --num_tests 10 --logname coverd-mnist-ConvSmall-t-5`|
| | ConvSmallPGD  | 6 | `python3 coverd.py --netname calzone_models/MNIST_ConvSmallPGD.onnx --dataset mnist --t 6 --timeout 18000  --gpu_num 8 --milp_num 5 --num_tests 10 --logname coverd-mnist-ConvSmallPGD-t-6`|
| F-MNIST | ConvSmallPGD  | 5 | `python3 coverd.py --netname calzone_models/FASHION_ConvSmallPGD.onnx --dataset fashion --t 5 --timeout 18000  --gpu_num 8 --milp_num 5 --num_tests 10 --logname coverd-fashion-ConvSmallPGD-t-5`|
