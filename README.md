# This project is mainly based on Segment Everything Everywhere All at Once: https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once and Segment Anything Model: https://github.com/facebookresearch/segment-anything

## SAM+SEEM for high quality automatic semantic segmentation


This project is based on two state of the art promptable segmentation models: **S**egment **A**nything **M**odel, [**SAM**](https://arxiv.org/abs/2304.02643) [[1]](#1), and **S**egment **E**verything **E**verywhere with **M**ultimodal Prompts, [**SEEM**](https://arxiv.org/abs/2304.06718) [[2]](#2). The main idea is to leverage SAM's ability to generate high quality masks, without a label, which will then be used as prompts for SEEM which will in its turn label them.

## Environment setup

I have used a conda environment, the environment and libraries setup is pretty straightforward.

If you're using anaconda the steps are as follows (after cloning this project):

```sh 
conda create -n seem_sam python=3.10.11
conda activate seem_sam
```
Now that you have created your virtual environment and activated it, it's time to install all the libraries and download the model weights in the corresponding folders.

### Installing the requirements
```sh
cd pipeline-semantic-segmentation
pip install -r requirements1.txt
pip install -r requirements2.txt
cd segment_anything
pip install -e .
```
### Downloading the weights
```sh
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ..
wget https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v1.pt
```
Now you're ready to go.

##  Main scripts 

### Classes definition
The classes are defined in the `pipeline-semantic-segmentation/demo_code/utils/constants.py` where you can notice two sets of classes_cmap and cmap dictionaries. Each pair is for a different number of classes: the first for the original 28 classes before merging the classes where there is confusion and disregarding some classes. And the other is for 19 classes after merging and disregarding some of them. You can alternate between depending if you have merged the classes or not (using the ```final_merging()``` function defined in ```pipeline-semantic-segmentation/demo_code/utils/util.py```). The pipeline originally generates the semantic segmentation of the image using the 28 classes, if you wish to merge the classes and disregard the badly matched ones, use the `final_mapping()` function.

In addition to that you can see the mapping from **COCO** classes to our classes in the `constants.py` as well.

### Image segmentation
The main script to generate mass annotations is `SEEMM_SAM/demo_code/inference_total.py`, modify it to save the pipeline segmentation, seem segmentation and seem logits under a specific path. A second script, `pipeline-semantic-segmentation/demo_code/inference_parallel.py`, is used to parallelize the process by dividing the total images into multiple batches and launch them in parallel as multiple processes on different GPUs. It basically launches the `inference_total.py` script many times but with different batches of images.

After setting the corresponding set of images in `inference_total.py`, you can launch a single process using: 
```sh
CUDA_VISIBLE_DEVICES=0 python inference_total.py
```
I have to mention that it is very important that the set of images is designated with a list of strings of image paths as they are saved.

To divide the images into multiple batches using multiple processes, you can use:
```sh
python inference_parallel
```
Before that, for the multiple processes, you need to specify the parameters which are the number of batches you want to divide the images into, as well as each GPU for each process. This is done in the `inference_parallel.py` script where you define the runs as follows. 

In `inference_parallel.py`:
```python
    runs = [
        (["--batch", "0", "--nb_batches", "8"], "0"),
        (["--batch", "1", "--nb_batches", "8"], "0"),
        (["--batch", "2", "--nb_batches", "8"], "1"),
        (["--batch", "3", "--nb_batches", "8"], "1"),
        (["--batch", "4", "--nb_batches", "8"], "2"),
        (["--batch", "5", "--nb_batches", "8"], "2"),
        (["--batch", "6", "--nb_batches", "8"], "3"),
        (["--batch", "7", "--nb_batches", "8"], "3"),
    ]
```
The first parameter ``--batch`` being the number of the bach for the process (usually it just an increasing number from 0 to the total number of processes you want to create). The second parameter ``--nb_batches`` is the total number of batches you want to divide the images into, so it stays the same for all the runs. As for the last parameter `(0,1,2,3)` in the last column, it's the GPU number on which you want to launch the process. Usually on 1 GPU of 24GB (NVIDIA RTX 3090), you can launch two of these processes in parallel since each use about 10GB at peak.

## Main functions

The main functions that were created, which are the functions that generate the SAM masks and then label them,

```python
def sam_inference(image, sam_mask_generator)
    #. . .
    return sam_masks

def seem_inference(image, sam_masks)
    #. . .
    return semantic_segmentation, . . .
```
can be found in `pipeline-semantic-segmentation/demo_code/utils/util.py`. If you wish to experiment with other combinations from filtering of the masks to filling the unsegmented areas to strategies to label the masks etc. these are the functions to modify.

## Jupyter notebooks

`pipeline-semantic-segmentation/demo_code/example.ipynb` shows an example of an image segmented by the pipeline. This includes all the steps: from generating the masks using SAM to labeling the full image.

<a id="1">[1]</a>:

**Segment Anything**
    - **Paper**: [**Segment Anything**](https://arxiv.org/abs/2304.02643)
    - **Authors**: Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick
    - **Journal**: arXiv:2304.02643
    - **Year**: 2023

<a id="2">[2]</a>:

**Segment Everything Everywhere with Multimodal Prompts**
    - **Paper**: [**Segment Everything Everywhere All at Once**](https://arxiv.org/abs/2304.06718)
    - **Authors**: Xueyan Zou, Jianwei Yang, Hao Zhang, Feng Li, Linjie Li, Jianfeng Wang, Lijuan Wang, Jianfeng Gao, Yong Jae Lee
    - **Journal**: arXiv:2304.06718
    - **Year**: 2023