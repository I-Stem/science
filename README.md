# science

This is the repo for I-Stem's science logic for our document and audio/video accessibility services.

# Contents
* [Introduction](#introduction)
* [Getting started](#getting-started)
* [Project structure](#project-structure)
* [Document accessibility](#document-accessibility)
    * [Layout analysis and recognition](#layout-analysis)
    * [Formatting](#layout-formatting)
* [Audio/video accessibility](#av-accessibility)
* [API specification](#api-spec)

# Introduction

A significant number of documents and media produced or shared online is inaccessible to people with disabilities. Often this content is scanned as images making it inaccessible, contains visual content or is not captioned, thereby excluding people who are blind or visually impaired, deaf or hard of hearing or those with some learning disabilities such as dyslexia. Through I-Stem's portal, our vision is to convert this inaccessible content into accessible formats. This repo has the science/algorithmic logic for our various workflows.

# Getting started

The code in this repo has been written in Python 3 exposed as Flask APIs which is in turn called by our I-Stem web app. To get started, we recommend creating a new virtual environment and then running pip install -r requirements.txt. 

# Project structure

The repo has been divided into a few directories.

* evaluation: This contains our evaluation logic to test our model performance as well as visualize the difference between original documents and those generated by our document accessibility models to understand specific gaps.
* layout_detection: This is the root directory for all of our work.
    * Config: This contains sample config files for deployment.
    * layout: This contains the logic for our general OCR/text extraction as well as document analysis (e.g. detection of tables, lists, headings, one/two-column layouts etc.) and their appropriate handling. This also contains our recreation script which takes these individual components and puts them all together to create an accessible document. The output from this flow is a json that contains all of the metadata (e.g. formatting information, object type such as table, paragraph etc.).
    * layout_formatting: Once we have the json output from layout, this directory contains logic to take it and generate the requested file format (docx, HTML, txt and MP3). This decomposition is deliberate to avoid multiple OCR/analysis calls when just trying to generate same file in multiple formats.
    * video_captioning: This directory contains our captioning and text extraction logic including training custom language models to handle domain-specific captioning.

# Document accessibility

## Layout analysis and recognition

This section describes the overall layout analysis/recognition flow. For specific details, please refer to individual files. All files referred here are in the "layout" directory.

* layout_ocr.py: This serves as a driver program taking in the input file, determining the file format, extracting images if a PDF, performing basic OCR/text extraction on it, sending it to our layout analysis models to understand the nature of the document and identify layout elements (e.g. headings, tables etc.), handling each of these layout elements using a combination of AI and rule-based approaches and finally recreating the document.
* ocr.py: This performs text extraction from images without any layout information. This takes in an image or a PDF and returns the text with their word and line level bounding boxes. Sample input and output is included as comments in the file.
* layout_inference.py: Using a faster RCNN ([click here for details](https://ieeexplore.ieee.org/abstract/document/8298720)), this takes in the image and returns bounding boxes with specific labels indicating the layout elements (text, image, table, title, list). Sample input and output is included as comments in the file.
* recreation.py: This takes in the output from the layout detection model described above, and then recreates the file using individual components. This also checks if the page is one column or two columns, thereby ensuring the correct reading order.
* Other files are responsible for handling specific types of layout elements such as tables (table_extraction.py), lists (list_handling.py) math (math_ocr.py).

## Formatting

This contains logic to take the json containing the text and the metadata obtained from text extraction and analysis, and converting that into specific file formats.

* docx_parser.py: for conversion to a Microsoft Word file.
* html_parser.py: to convert to an HTML file.
* save_file.py: To save the file to cloud storage.
* tts_parser.py: For MP3 generation by leveraging text-to-speech.
* utils.py: To handle reusable code (currently specifically around lists) during the conversion process.

# Audio/video accessibility

this section describes the A/V accessibility flow.

* All of the captioning and text extraction code currently lives in main.py file in the "video_captioning" directory.
* The captioning process consists of uploading the video, training an optional custom language model using textual content related to the video being captioned (in which case we use the document accessibility flow described above to get the txt output first followed by training a custom language model) and captioning (with or without the custom language model). 
* The text extraction process extracts keyframes, detects text and uses OCR to extract the text.
* For specific details, refer to the file.

# API specification

## Document analysis and recognition

-   **Endpoint: /api/v1/ocr
-   ** Input: doc_type (MATH or NONMATH; default is NONMATH), hash (used to avoid processing if a file has already been processed) and the file uploaded by the user
-   **Output: If successful, the API returns immediately with message and status. Once the processing is completed, a callback is sent which contains the actual json containing the document.

## Formatting

-   **Endpoint: /api/v1/ocr/format
-   ** Input: json (json obtained from the first step), format (desired file format; valid options include "DOCX", "HTML", "TXT", "PDF" and "MP3") hash (used to avoid processing if a file has already been processed) documentName (used for rendering in the app and storing), 
-   **Output: status, message, url (generated file URL)

## Uploading video

-   **Endpoint: /api/v1/vc
-   ** Input: name (name of the video file), url (URL of the video), hash (used to avoid processing if a file has already been processed), languageModelId (optional language model ID)
-   **Output: If upload is successful, the API returns immediately with error (false in case of success, true otherwise), message and videoId. Once the processing is complete, the callback URL is called.

## Video allback

-   **Endpoint: /api/v1/vc/callback
-   ** Input: documentName (name of the video document), id (video ID obtained from the upload API), hash (used to avoid processing if a file has already been processed), type (request type; valid options include "CAPTION" for captions only, "OCR" for text extraction only and "OCR_CAPTION" for both text extraction and captions), outputFormat (output format for captions; valid values include "txt" and "srt")
-   **Output: url, hash, duration in case of a success, error and message in case of failure

## Training custom language model

-   **Endpoint: /api/v1/customspeech
-   ** Input: name (name of the model), fileName (name of the file being used for training), fileUrl (URL of the file being used for training), 
-   **Output: error, message, languageModelId (if successful)
