# EVALUATION  
## OCR  
ocr_eval_metrics.ipynb -  Combines individual file results and calculates overall metrics for the dataset.  
html - Contains the html output files.   
text_comparison - Text alignments (OCR text vs Ground truth).  
results - Individual file results. 
### Current Metrics  
This section reflects the current metrics of our AI model and are being continuously updated as we make progress.  
Average text accuracy = 91.74%  
Average F1 score = 0.87  
Average table text accuracy=  90.84%  
Total files used =  68  
Ground truths can be found at - https://github.com/I-Stem/ocr_dataset  
  
## SPEECH  
### Current Metrics  
Word Error Rate - 8.67  
### Dataset  
An ensemble of open source, proprietary and custom created datasets were used to evaluate the speech models. Links for available datasets have been given below and reach out to info@inclusivestem.org to access our datasets.  
* https://commonvoice.mozilla.org/en/datasets
* https://www.iitm.ac.in/donlab/tts/database.php
* https://www.kaggle.com/rtatman/speech-accent-archive
### Observations
* Performance on the US accent was the best, followed by UK and Indian accents.
* Performance on clear audio files was better than on noisy data even after our denoising attempts.
* The error rate was high because of the difference in representation of numbers (cardinal form vs number form).
* Punctuation marks were also very hard to predict.
