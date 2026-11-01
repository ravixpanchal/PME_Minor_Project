"""Keep observable signature ink outside printed text; never reconstruct strokes."""
import math

import numpy as np
from PIL import Image, ImageDraw


def isolate_signature(signature, text_boxes, stamp_in_signature=None):
    if not isinstance(text_boxes, list):
        raise ValueError('Signature text_boxes must be an explicit list')
    boxes = list(text_boxes)
    if stamp_in_signature is not None:
        boxes.append(stamp_in_signature)
    mask = Image.new('L', signature.size, 255)
    draw = ImageDraw.Draw(mask)
    for box in boxes:
        if not isinstance(box,list) or len(box)!=4 or any(type(v) not in (float,int) or not math.isfinite(v) for v in box):
            raise ValueError('Invalid signature text mask box')
        x1,y1,x2,y2 = box
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError('Text mask coordinates must be normalized 0..1')
        draw.rectangle((max(0,round(x1*signature.width)-2),max(0,round(y1*signature.height)-2),
                        min(signature.width,round(x2*signature.width)+2),
                        min(signature.height,round(y2*signature.height)+2)),fill=0)
    cleaned = Image.composite(signature.convert('RGB'),Image.new('RGB',signature.size,'white'),mask)
    ink = np.asarray(signature.convert('L')) < 180
    remaining = ink & (np.asarray(mask)>0)
    # Persist the exclusion mask as alpha in the matching-only PNG. The raw
    # evidence remains RGB and unchanged.
    cleaned.putalpha(mask)
    fraction = float(remaining.sum()/max(1,ink.sum()))
    return cleaned, mask, fraction


def overlapping_stamp(signature_box, stamp_box):
    if not stamp_box:
        return None
    x1,y1,x2,y2 = signature_box
    a,b,c,d = stamp_box
    left,top,right,bottom = max(x1,a),max(y1,b),min(x2,c),min(y2,d)
    if left>=right or top>=bottom:
        return None
    return [(left-x1)/(x2-x1),(top-y1)/(y2-y1),(right-x1)/(x2-x1),(bottom-y1)/(y2-y1)]
