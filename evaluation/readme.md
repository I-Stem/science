# EVALUATION  
## OCR  
ocr_eval_metrics.ipynb -  Combines individual file results and calculates overall metrics for the dataset.  
html - Contains the html output files.   
text_comparison - Text alignments (OCR text vs Ground truth).  
results - Individual file results. 
### Current Metrics ( As of 28 Jun 2021)
This section reflects the current metrics of our AI model and are being continuously updated as we make progress.  
Average text accuracy = 96.2%  
Average F1 score = 0.885  
Average table text accuracy=  91.29%  
Total files used =  64  
Ground truths can be found at - https://github.com/I-Stem/ocr_dataset  
### Datasets  
* https://github.com/ibm-aur-nlp/PubLayNet
* https://www.primaresearch.org/dataset/
* https://github.com/uchidalab/GTDB-Dataset
  
## SPEECH  
### Current Metrics  
Word Error Rate - 8.67  
### Dataset  
An ensemble of open source, proprietary and custom created datasets were used to evaluate the speech models. Links for available datasets have been given below and reach out to info@inclusivestem.org to access our datasets (we were unable to open source the created dataset due to licensing issues).  
* https://commonvoice.mozilla.org/en/datasets
* https://www.iitm.ac.in/donlab/tts/database.php
* https://www.kaggle.com/rtatman/speech-accent-archive
### Observations
* Performance on the US accent was the best, followed by UK and Indian accents.
* Performance on clear audio files was better than on noisy data even after our denoising attempts.
* The error rate was high because of the difference in representation of numbers (cardinal form vs number form).
* Punctuation marks were also very hard to predict.
* Using custom speech models improved the accuracy by an average of 2.8%, although the limited number of samples (with each sample consisting of audio, ground truth and related textual material) did not allow us to do as thorough of an evaluation we would have liked to. We continue to find and create more relevant dataset to evaluate this more comprehensively.
