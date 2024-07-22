"""Functions and data structures for dealing with performance indicators"""

import dataclasses

from typing import Union


KNOWN_INDICATORS = dict()


@dataclasses.dataclass(eq=True, frozen=True)
class Indicator:
    """Description of a performance indicator

    Attributes
    ----------
    name : str
        Name of the quality indicator. Must match the column name
        in the Results object.        
    larger_is_better : bool, default = True
        True if larger values of the indicator are better, False 
        otherwise.    
    """
    name: str
    larger_is_better: bool = True

    
def register(ind: Indicator):
    """Register a new performance indicator

    cocoviz has a global list of know performance indicators with their
    associated metadata. Using this function, you can add additional indicators
    so that you do not have to specify their properties every time.

    Parameters
    ----------
    ind : Indicator
        Indicator to add to list of known indicators
    """
    KNOWN_INDICATORS[ind.name] = ind


def deregister(ind: Union[Indicator, str]):
    """_summary_

    Parameters
    ----------
    ind : Indicator or str
        Indicator to remove from list of known indicators

    Raises
    ------
    NotImplementedError
        when `ind` is neither a string nor an instance of Indicator
    """
    if isinstance(ind, str):
        del KNOWN_INDICATORS[ind]
    elif isinstance(ind, Indicator):    
        del KNOWN_INDICATORS[ind.name]
    else:
        raise NotImplementedError()


## Register some common and not so common quality indicators
register(Indicator("hypervolume", larger_is_better=True))
register(Indicator("hv", larger_is_better=True))
register(Indicator("r2", larger_is_better=True))
register(Indicator("time", larger_is_better=False))
