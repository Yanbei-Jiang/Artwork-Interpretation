# KALE: An Artwork Image Captioning System Augmented with Heterogeneous Graph

## Introduction
Exploring the narratives conveyed by fine-art paintings is a challenge in image captioning, where the goal is to generate descriptions that not only precisely represent the visual content but also offer a in-depth interpretation of the artwork's meaning. The task is particularly complex for artwork images due to their diverse interpretations and varied aesthetic principles across different artistic schools and styles. In response to this, we present KALE (Knowledge-Augmented vision-Language model for artwork Elaborations), a novel approach that enhances existing vision-language models by integrating artwork metadata as additional knowledge. KALE incorporates the metadata in two ways: firstly as direct textual input, and secondly through a multimodal heterogeneous knowledge graph. To optimize the learning of graph representations, we introduce a new cross-modal alignment loss that maximizes the similarity between the image and its corresponding metadata. Experimental results demonstrate that KALE achieves strong performance over existing state-of-the-art work across several artwork datasets, particularly in achieving impressive CIDEr scores.

<img src="figures/model_architecture1.png" width="600"> 
<img src="figures/model_architecture2.png" width="600"> 


## Setup


### Requirements
* [PyTorch](https://pytorch.org/) version == 1.11.0
* numpy == 1.23.5
* python == 3.9.16
* torchvision == 0.12.0
* torch-scatter == 2.0.9
* torch-geometric == 2.4.0
* wandb

### File and Package Required
1. Download pre-trained vision transformer [vit-l-14](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/mPLUG/ViT-L-14.tar). (Do not UNZIP it!!)
2. Download language evaluation tool and unzip it to the main directory [language_evaluation.zip](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/mPLUG/language_evaluation.tar).
3. Download bert_base_uncased and unzip it to the main directory [bert_base_uncased.zip](https://drive.google.com/drive/folders/1m-STuGx1U7c8tQ_KkqOv39CGaFTAs6WY?usp=drive_link).
4. Download mPLUG pre-trained checkpoints [mplug_large_v2.pth](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/mPLUG/mplug_large_v2.pth).    
5. Download heterogeneous graph with pre-trained embeddings [graph.pkl](https://drive.google.com/drive/folders/1m-STuGx1U7c8tQ_KkqOv39CGaFTAs6WY?usp=drive_link).
6. Download pre-trained FastText and unzip it to the main directory [cc.en.300.bin](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz).
7. Download dataset annotation files and unzip it to the main directory [data.zip](https://drive.google.com/drive/folders/1m-STuGx1U7c8tQ_KkqOv39CGaFTAs6WY?usp=drive_link).

### Dataset Images Required
1. Artpedia: Download Artpedia dataset and unzip the 'images' folder to the directory 'data/Artpedia/' [artpedia.zip](https://aimagelab.ing.unimore.it/imagelab/uploadedFiles/artpedia.zip).
2. Artcaps: Download Artcaps images and unzip the 'images' folder to the directory 'data/ArtCap/' [images.zip](https://drive.google.com/drive/folders/1l8NK8mvpkG3UfrTuTNyBZciIjFquV2ia?usp=sharing).
3. SemArt: Download SemArt dataset and unzip the 'Images' folder to the directory 'data/SemArt/' [semart.zip](https://astondr-prod.leaf.cosector.com/id/eprint/380/1/SemArt.zip).

### Training
For ArtCap dataset:
1. KALE (without metadata): Run ```python KALE_artcap.py```

For Artpedia dataset:
1. KALE (without metadata): Run ```python KALE_artpedia.py```
2. KALE (with metadata): Run ```python KALE_metadata_artpedia.py```

For SemArt v1.0 (Visual/Contextual Split) datasets:
1. KALE (without metadata): Run ```python KALE_semart1.py```
2. KALE (with metadata): Run ```python KALE_metadata_semart1.py```

For SemArt v2.0 (Form/Content/Context Split) datasets:
1. KALE (without metadata): Run ```python KALE_semart2.py```
2. KALE (with metadata): Run ```python KALE_metadata_semart2.py```


## Generated Examples
<img src="figures/examples.png">




