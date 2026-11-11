"""Single-class, confidence-ordered plate matching before application caps."""
from __future__ import annotations
import math


def _box(value):
    if len(value)!=4 or not all(math.isfinite(float(x)) for x in value):
        raise ValueError("box requires four finite coordinates")
    x0,y0,x1,y1=map(float,value)
    if x1<=x0 or y1<=y0:
        raise ValueError("box must have positive area")
    return x0,y0,x1,y1


def box_iou(first,second):
    a,b=_box(first),_box(second)
    intersection=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    return intersection/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)


def match_plate_boxes(predictions,truth,*,threshold=.5):
    """Match sorted predictions once per GT; count extra matching boxes as FP.

    Input predictions must already be ordered by decreasing detector confidence.
    Scores are not fused with OCR or interpreted as end-to-end accuracy.
    """
    if not math.isfinite(threshold) or not 0<threshold<=1:
        raise ValueError("invalid matching threshold")
    predictions=[_box(b) for b in predictions];truth=[_box(b) for b in truth]
    matched=set();duplicates=other_false=0
    for prediction in predictions:
        overlaps=[box_iou(prediction,target) for target in truth]
        eligible=[i for i,overlap in enumerate(overlaps) if overlap>=threshold]
        available=[i for i in eligible if i not in matched]
        if available:
            matched.add(max(available,key=lambda i:overlaps[i]))
        elif eligible:
            duplicates+=1
        else:
            other_false+=1
    return {"tp":len(matched),"fp":duplicates+other_false,"fn":len(truth)-len(matched),
            "duplicate_boxes":duplicates,"other_false_boxes":other_false}
