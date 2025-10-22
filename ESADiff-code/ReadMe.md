## Edge-Aware Diffusion Segmentation Model with Hessian Priors for Automated Diaphragm Thickness Measurement in Ultrasound Imaging ([IEEE Xplore](https://ieeexplore.ieee.org/document/11134285)(Early Access))
[Chen-long Miao](https://github.com/18952171773); Yikang He; Baike Shi; Zhongkai Bian; Wenxue Yu; Yang Chen; Guang-Quan Zhou

# News
- We have uploaded the initial code, and the detailed, organized code will be gradually uploaded later.

## I. Before Starting.
1. install torch
~~~
conda create -n esadiff python=3.9
conda activate esadiff
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
~~~
2. install other packages.
~~~
pip install -r requirement.txt
~~~


## II. Prepare Data.
The training data structure should look like:
```commandline
|-- $data_root
|   |-- image
|   |-- |-- raw
|   |-- |-- |-- XXXXX.png
|   |-- |-- |-- XXXXX.png
|   |-- edge
|   |-- |-- raw
|   |-- |-- |-- XXXXX.png
|   |-- |-- |-- XXXXX.png
```
The testing data structure should look like:
```commandline
|-- $data_root
|   |-- XXXXX.png
|   |-- XXXXX.png
```


## III. Training.
1. train the first stage model (AutoEncoder):
python train_vae.py --cfg ./configs/first_stage.yaml
~~~
2. you should add the final model weight of the first stage to the config file `./configs/ESADiff_train.yaml` (**line 42**), then train latent diffusion-edge model:
~~~
python train_cond_ldm.py --cfg ./configs/ESADiff_train.yaml
~~~

## IV. Inference.
make sure your model weight path is added in the config file `./configs/ESADiff_sample.yaml` (**line 73**), and run:
~~~
python sample_cond_ldm.py --cfg ./configs/ESADiff_sample.yaml
~~~
Note that you can modify the `sampling_timesteps` (**line 11**) to control the inference speed.

## Thanks
Thanks to the base code [DDM-Public](https://github.com/GuHuangAI/DDM-Public) and [DiffusionEdge](https://github.com/GuHuangAI/DiffusionEdge)
