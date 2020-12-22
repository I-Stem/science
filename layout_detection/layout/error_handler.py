# define Python user-defined exceptions for all the classes


class Error(Exception):
    """Base class for other exceptions"""
    pass

class OCRError(Error):
    """OCR api error"""
    pass


class TableExtractionError(Error):
    """Table extraction error"""
    pass


class LayoutExtractionError(Error):
    """Layout extraction error"""
    pass

class ImageRecreationError(Error):
    """Image recreaction error"""
    pass

class FormatingError(Error):
    """Word doc formating error"""
    pass