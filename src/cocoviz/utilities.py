import colorsys
from typing import List


def scale_lightness(rgb: List[float], scale_l: float):
    """Scale lightness of an RGB color"""
    hue, lightness, saturation = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(hue, min(1, lightness * scale_l), saturation)
