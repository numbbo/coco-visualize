class NoSuchIndicatorException(Exception):
    """Raised when an unknown indicator is accessed"""

    pass


class IndicatorMismatchException(Exception):
    """Raised when indicators don't match between results"""

    pass


class BadResultSetException(Exception):
    """Raised when a runtime profile doesn't make sense for a result set"""    
    
    pass
