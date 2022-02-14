# Badges

- [![Documentation Status](https://readthedocs.org/projects/science/badge/?version=latest)](https://science.readthedocs.io/en/latest/?badge=latest)

# science

This is the repo for I-Stem's science logic for our document and audio/video accessibility services.

# Contents
* [Introduction](#introduction)
* [Getting started](#getting-started)
* [Project structure](#project-structure)
* [Document accessibility](#document-accessibility)
    * [Layout analysis and recognition](#Layout-analysis-and-recognition)
    * [Model architecture](#Model-architecture)
    * [Data](#Data)
    * [Weights and Parameters](#Weights-and-Parameters)
    * [Platforms](#Platforms)
    * [Formatting](#Formatting)
    * [Evaluation](#Evaluation)
* [Audio/video accessibility](#av-accessibility)
* [API specification](#api-spec)

# Introduction

A significant number of documents and media produced or shared online is inaccessible to people with disabilities. Often this content is scanned as images making it inaccessible, contains visual content or is not captioned, thereby excluding people who are blind or visually impaired, deaf or hard of hearing or those with some learning disabilities such as dyslexia. Through I-Stem's portal, our vision is to convert this inaccessible content into accessible formats. This repo has the science/algorithmic logic for our various workflows.

   To know more about who we are and what we aim to develop, go [here](https://i-stem.github.io).

# Getting started

The code in this repo has been written in Python 3 exposed as Flask APIs which is in turn called by our I-Stem web app. To get started, we recommend creating a new virtual environment and then running pip install -r requirements.txt. 

# Project structure

The repo has been divided into a few directories.

* evaluation: This contains our evaluation logic to test our model performance as well as visualize the difference between original documents and those generated by our document accessibility models to understand specific gaps.
* layout_detection: This is the root directory for all of our work.
    * Config: This contains sample config files for deployment.
    * layout: This contains the logic for our general OCR/text extraction as well as document analysis (e.g. detection of tables, lists, headings, one/two-column layouts etc.) and their appropriate handling. This also contains our recreation script which takes these individual components and puts them all together to create an accessible document. The output from this flow is a json that contains all of the metadata (e.g. formatting information, object type such as table, paragraph etc.).
    * layout_formatting: Once we have the json output from layout, this directory contains logic to take it and generate the requested file format (docx, HTML, txt and MP3). This decomposition is deliberate to avoid multiple OCR/analysis calls when just trying to generate the same file in multiple formats.
    * video_captioning: This directory contains our captioning and text extraction logic including training custom language models to handle domain-specific captioning.

# Document accessibility

## Layout analysis and recognition

This section describes the overall layout analysis/recognition flow. For specific details, please refer to individual files. All files referred here are in the "layout" directory.

* layout_ocr.py: This serves as a driver program taking in the input file, determining the file format, extracting images if a PDF, performing basic OCR/text extraction on it, sending it to our layout analysis models to understand the nature of the document and identify layout elements (e.g. headings, tables etc.), handling each of these layout elements using a combination of AI and rule-based approaches and finally recreating the document.
* ocr.py: This performs text extraction from images without any layout information. This takes in an image or a PDF and returns the text with their word and line level bounding boxes. Sample input and output is included as comments in the file.
* layout_inference.py: Using a faster RCNN ([click here for details](https://ieeexplore.ieee.org/abstract/document/8298720)), this takes in the image and returns bounding boxes with specific labels indicating the layout elements (text, image, table, title, list). Sample input and output is included as comments in the file.
* layout_Secondary.py:  In case of fallback or poor performance from layout_inference we leverage Detectron2([click here for details](https://ai.facebook.com/blog/-detectron2-a-pytorch-based-modular-object-detection-library-/)) using Mask RCNN ([click here for details](https://github.com/matterport/Mask_RCNN)).
* recreation.py: This takes in the output from the layout detection model described above, and then recreates the file using individual components. This also checks if the page is one column or two columns, thereby ensuring the correct reading order.
* Other files are responsible for handling specific types of layout elements such as tables (table_extraction.py), lists (list_handling.py) math (math_ocr.py).

## Model architecture

Mask R-CNN is a state of the art model for instance segmentation, developed on top of Faster R-CNN. Faster R-CNN is a region-based convolutional neural network, that returns bounding boxes for each object and its class label with a confidence score. 
### The architecture of Faster R-CNN
![The architecture of Faster R-CNN](https://www.researchgate.net/profile/Zhipeng-Deng-2/publication/324903264/figure/fig2/AS:640145124499471@1529633899620/The-architecture-of-Faster-R-CNN.png)  
Credits : Multi-scale object detection in remote sensing imagery with convolutional neural networks - Scientific Figure on ResearchGate. Available from: https://www.researchgate.net/figure/The-architecture-of-Faster-R-CNN_fig2_324903264

Faster R-CNN predicts object class and bounding boxes. Mask R-CNN is an extension of Faster R-CNN with additional branch for predicting segmentation masks on each Region of Interest (RoI) 
### The overall network architecture of Mask R-CNN
![The overall network architecture of Mask R-CNN](https://www.researchgate.net/publication/336615317/figure/fig1/AS:815040580042752@1571332225271/The-overall-network-architecture-of-Mask-R-CNN.png)  
Credits : An automatic nuclei segmentation method based on deep convolutional neural networks for histopathology images - Scientific Figure on ResearchGate. Available from: https://www.researchgate.net/figure/The-overall-network-architecture-of-Mask-R-CNN_fig1_336615317

## Data

A combination of multiple datasets was used to train and fine-tune the above mentioned models and evaluate their performance, some of them were publically available like - 
### PubLayNet
PubLayNet is a large dataset of document images from PubMed Central Open Access Subset. Each document’s layout is annotated with both bounding boxes and polygonal segmentations. Refer to this link ([link](https://github.com/ibm-aur-nlp/PubLayNet)) to read more about the dataset and accessing it.
### PRImA Layout Analysis Dataset
This dataset has been created primarily for the evaluation of layout analysis (physical and logical) methods. It contains realistic documents with a wide variety of layouts, reflecting the various challenges in layout analysis.
Refer to this link ([link](https://www.primaresearch.org/dataset/)) to read more about the dataset and accessing it.  

These two datasets were mostly used for pre-training, in addition to them we came up with our own dataset for fine-tuning the models to our specfic use case. Consisting of thousands of pages, from a variety of input sources covering academic papers, handouts, text books, notes etc.  However, since this dataset contains some images that are not in the public domain, we only have permission to use and share them for research purposes. Please contact us if you would like to access that dataset with appropriate justification by sending us an email to info@inclusivestem.org .

## Weights and Parameters

The model weights and the configuration files for pre-trained Faster-RCNN and Mask-RCNN models on PubLayNet can be downloaded from ([link](https://github.com/ibm-aur-nlp/PubLayNet/tree/master/pre-trained-models/)).

The config file for the prima trained mask rcnn model can be downloaded from ([link](https://www.dropbox.com/s/yc92x97k50abynt/config.yaml?dl=1))


## Platforms

Cloud hosting platform, AWS was used for deployment of the code base and the deep learning models. Dedicated Amazon EC2 servers were used to host the apis, they in turn rely on sagemaker servers on which the models were deployed to.  
Amazon Elastic Compute Cloud (EC2) is a part of Amazon Web Services (AWS), that allows users to rent virtual computers on which to run their own computer applications. Memory optimized server were chosen for this ([EC2r5](https://aws.amazon.com/ec2/instance-types/r5/)).
Amazon SageMaker is a fully-managed service that enables easy build, train, and deployment of machine learning models at any scale. Compute optimized server were chosen for this ([EC2c5](https://www.apptio.com/blog/aws-ec2-c5-instances/)).

## Formatting

This contains logic to take the json containing the text and the metadata obtained from text extraction and analysis, and converting that into specific file formats.

* docx_parser.py: for conversion to a Microsoft Word file.
* html_parser.py: to convert to an HTML file.
* save_file.py: To save the file to cloud storage.
* tts_parser.py: For MP3 generation by leveraging text-to-speech.
* utils.py: To handle reusable code (currently specifically around lists) during the conversion process.

## Evaluation 
### IOU
Intersection over Union is an evaluation metric used to measure the accuracy of an object detector on a particular dataset
In the numerator we compute the area of overlap between the predicted bounding box and the ground-truth bounding box. The denominator is the area of union, or more simply, the area encompassed by both the predicted bounding box and the ground-truth bounding box. Dividing the area of overlap by the area of union yields our final score — the Intersection over Union.
![Intersection over Union calculation diagram](https://www.researchgate.net/profile/Rafael-Padilla/publication/343194514/figure/fig2/AS:916944999956482@1595628132920/Intersection-Over-Union-IOU.ppm)  
More information on evaluation and our performance metrics is provided in the evaluation folder and the dataset used for evaluation can be found at ([link](https://github.com/I-Stem/ocr_dataset/))

# Audio/video accessibility

This section describes the A/V accessibility flow.

* All of the captioning and text extraction code currently lives in main.py file in the "video_captioning" directory.
* The captioning process consists of uploading the video, training an optional custom language model using textual content related to the video being captioned (in which case we use the document accessibility flow described above to get the txt output first followed by training a custom language model) and captioning (with or without the custom language model). 
* The text extraction process extracts keyframes, detects text and uses OCR to extract the text.
* For specific details, refer to the file.

# API specification

Please have a look [here](https://i-stem.github.io/science).
