"""Compatibility shim: re-export get_dataloaders from tft.data.loaders.

保留旧导入路径：from data.load_dataset import get_dataloaders
"""
from tft.data.loaders import get_dataloaders  # noqa: F401
