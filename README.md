# FluxGAN

This repository contains collection of various Generative Adverserial Models implemented with Flux. Model architectures may not be exact as those proposed in the paper but key focus have been put on getting the core ideas rather than just copying the model architectures layer by layer. Any suggestions and contributions are always welcome.

## Usage
In order to the run (train) the models implemented in this repo, first you need to clone the repo. After that, use the following code to activate the environment
```
using Pkg; Pkg.activate("."); Pkg.instantiate();
```
Then simply go to the directory of the model you want to try, and run the .jl file. The model will start training and once completed you can see the results in the output folder which will be created when you run the model.

# Roadmap
## Model implemented so Far
* AAE
* ACGAN
* CGAN
* CoupledGAN
* InfoGAN
* LSGAN
* SGAN
* VanillaGAN
* WGAN

## What next
* Update the model versions to be compatible with Flux version 11.0
* Implement few more popular GAN models in Julia, like
  * CycleGAN
  * ClusterGAN
  * DiscoGAN
  * Context Encoder
* Add Generated samples images for the models implemented here.
  
  
# Contact Details
In case you have any questions related to implementation of any of the models available here, feel free to tag me (@Adarsh Kumar) on Julia's slack or create an issue on this Github repository.


# References
<li> https://github.com/eriklindernoren/PyTorch-GAN.git
<li> https://github.com/eriklindernoren/Keras-GAN.git
<li> https://machinelearningmastery.com/semi-supervised-generative-adversarial-network/

