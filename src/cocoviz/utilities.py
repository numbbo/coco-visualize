import colorsys
from typing import List


def scale_lightness(rgb: List[float], scale_l: float):
    """Scale the lightness of a RGB color.

    Parameters
    ----------
    rgb : (float, float, float)
        Red, green and blue values of color.
    scale_l : float
        Scaling factor for the lightness. 
        If smaller than one, the result will be darker.
        If larger than one, the result will be lighter.

    Returns
    -------
    (float, float, float)
        Red, green and blue values of the lightened or darkened color.
    """
    hue, lightness, saturation = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(hue, min(1, lightness * scale_l), saturation)
