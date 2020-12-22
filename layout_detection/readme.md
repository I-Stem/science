## Getting Started

> _Note: Copy the .env file in the current folder and add your `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`_

```
pip install -r requirements.txt
python nltk_lib_install.py
=======
# To install pipenv (if it's already not installed)
pip install pipenv

# To start the shell and install dependencies
pipenv shell
pipenv install

# To start the application
python application.py
```

## Usage
```python
from layout.layout_ocr import LayoutOcr
img_path=""

# if you don't want to save, Then you will json as output.
LOcr = LayoutOcr( open(img_path, 'rb').read())
# LOcr = LayoutOcr( open(pdf_path, 'rb').read(), input_type="pdf")
result = LOcr.result

```

## Notes

1. Currently accepted file format for ocr: 'pdf', 'png', 'jpg', 'jpeg', 'bmp', 'tiff' 
2. poppler-utils binary is needed for page count of pdf files 
3. wkhtmltopdf binary is needed for html to pdf conversion
4. The code requires at least python 3.6
