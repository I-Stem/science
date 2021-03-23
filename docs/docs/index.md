---
id: index
title: API Documentation
slug: /
---


# API specification

For a basic understanding of the following API, go to repo home page [here](https://github.com/I-Stem/science).

## APIs for Document Accessibility

### Document analysis and recognition

-   **Endpoint: /api/v1/ocr
-   ** Input: doc_type (MATH or NONMATH; default is NONMATH), hash (used to avoid processing if a file has already been processed) and the file uploaded by the user
-   **Output: If successful, the API returns immediately with message and status. Once the processing is completed, a callback is sent which contains the actual json containing the document.

### Formatting

-   **Endpoint: /api/v1/ocr/format
-   ** Input: json (json obtained from the first step), format (desired file format; valid options include "DOCX", "HTML", "TXT", "PDF" and "MP3") hash (used to avoid processing if a file has already been processed) documentName (used for rendering in the app and storing), 
-   **Output: status, message, url (generated file URL)

## APIs for Video Accessibility

### Uploading video

-   **Endpoint: /api/v1/vc
-   ** Input: name (name of the video file), url (URL of the video), hash (used to avoid processing if a file has already been processed), languageModelId (optional language model ID)
-   **Output: If upload is successful, the API returns immediately with error (false in case of success, true otherwise), message and videoId. Once the processing is complete, the callback URL is called.

### Video Callback

-   **Endpoint: /api/v1/vc/callback
-   ** Input: documentName (name of the video document), id (video ID obtained from the upload API), hash (used to avoid processing if a file has already been processed), type (request type; valid options include "CAPTION" for captions only, "OCR" for text extraction only and "OCR_CAPTION" for both text extraction and captions), outputFormat (output format for captions; valid values include "txt" and "srt")
-   **Output: url, hash, duration in case of a success, error and message in case of failure

### Training custom language model

-   **Endpoint: /api/v1/customspeech
-   ** Input: name (name of the model), fileName (name of the file being used for training), fileUrl (URL of the file being used for training), 
-   **Output: error, message, languageModelId (if successful)
