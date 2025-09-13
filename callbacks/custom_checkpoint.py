"""Compatibility shim for callbacks.

Old path: callbacks.custom_checkpoint.CustomCheckpoint
New path: tft.callbacks.custom_checkpoint.CustomCheckpoint
"""
from tft.callbacks.custom_checkpoint import *  # noqa: F401,F403
