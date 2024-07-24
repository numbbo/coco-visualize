class NoSuchIndicatorException(Exception):
    """Raised when an unknown indicator is accessed"""

    pass


class IndicatorMismatchException(Exception):
    """Raised when indicators don't match between results"""

    pass


class BadRuntimeProfileException(Exception):
    """Raised when a runtime profile doesn't make sense for a result set"""    
    
    pass


class UnknownIndicatorException(Exception):
    """Raised when an indicator is passed as a string and hasn't been registered previously"""
    def __init__(self, name: str):
        from .indicator import KNOWN_INDICATORS
        super()
        self.add_note(f"""You passed the string "{name}" as an indicator that was not previously registered.
                      
To use a quality indicator, cocoviz needs to know if the indicator needs to be 
minimized or maximized. You can either pass in an instance of the Indicator 
class or register the indicator once with the `register()` function contained 
in `cocoviz.indicator`.

Currently registered indicators: "{'", "'.join(KNOWN_INDICATORS.keys())}"
                      
Example
-------

>>> from cocoviz import Indicator
>>> ind = Indicator("hypervolume", larger_is_better=True)
>>> runtime_profiles(results, indicator=ind)

or

>>> import cocoviz.indicator as ci
>>> ci.register(ci.Indicator("hypervolume", larger_is_better=True))
>>> runtime_profiles(results, "hypervolume")                                            
""")    