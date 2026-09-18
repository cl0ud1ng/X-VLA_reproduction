"""Project-local equivalent of RoboTwin's official decode_image_bit.

Source reference: RoboTwin@96c1fea ``data/decode_image_bit.py`` /
XPolicyLab@c37109c ``XPolicyLab.utils.process_data.decode_image_bit``.
The returned array is already RGB; callers must not apply BGR conversion.
"""
from __future__ import annotations
import cv2
import numpy as np

def _single(value):
    if isinstance(value,np.ndarray) and value.dtype.kind in {'S','U'}:
        value=value.item() if value.ndim==0 else value.tobytes()
    if isinstance(value,str): value=value.encode('utf-8')
    elif isinstance(value,memoryview): value=value.tobytes()
    if isinstance(value,(bytes,bytearray)): value=value.rstrip(b'\0')
    elif isinstance(value,np.ndarray): value=np.ascontiguousarray(value)
    image=cv2.imdecode(np.frombuffer(value,np.uint8),cv2.IMREAD_COLOR)
    if image is None: raise ValueError(f'failed to decode image bits (type={type(value).__name__})')
    return image

def decode_image_bit(image_bits):
    if isinstance(image_bits,(bytes,bytearray,memoryview,str)): return _single(image_bits)
    if isinstance(image_bits,np.ndarray):
        if image_bits.dtype.kind in {'S','U','O'}:
            if image_bits.ndim==0: return _single(image_bits.item())
            return np.stack([_single(x) for x in image_bits],axis=0)
        if image_bits.dtype==np.uint8:
            if image_bits.ndim==1: return _single(image_bits)
            if image_bits.ndim==2: return np.stack([_single(x) for x in image_bits],axis=0)
        return image_bits
    if isinstance(image_bits,(list,tuple)): return np.stack([decode_image_bit(x) for x in image_bits],axis=0)
    return _single(image_bits)
