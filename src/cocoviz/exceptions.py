class NoSuchIndicatorException(Exception):
    """Raised when an unknown indicator is accessed"""

    pass


class IndicatorMismatchException(Exception):
    """Raised when indicators don't match between results"""

    pass