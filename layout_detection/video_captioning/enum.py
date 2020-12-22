from enum import Enum

class RequestType(Enum):
    CAPTION = "CAPTION"
    OCR = "OCR"
    OCR_CAPTION = "OCR_CAPTION"

    def __str__(self):
        return self.value

class FileExtensions(Enum):
    TEXT = 'txt'
    ZIP = 'zip'
    DOCX = 'docx'
    PDF = 'pdf'
    HTML = 'html'

    def __str__(self):
        return self.value