"""Compatibility shim for model module.

Old path: model.tft_module.MyTFTModule
New path: tft.models.tft_module.MyTFTModule
"""
from tft.models.tft_module import *  # noqa: F401,F403
