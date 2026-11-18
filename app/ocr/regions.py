"""Geometry-only first text-pair proposals with source-coordinate provenance.

This localizer does not classify characters, recognize countries or replace
text. A proposal may contain digits or background and needs model validation.
"""
from dataclasses import dataclass
import math
import cv2
import numpy as np
from PIL import Image


@dataclass
class PrefixProposal:
    crop: Image.Image
    plate: Image.Image
    box_xyxy: list[int]
    source_polygon: list[list[float]]
    source_to_plate: list[list[float]]
    rectified: bool
    components: list[list[int]]


def plate_view(image):
    rgb = np.asarray(image.convert('RGB'))
    height,width = rgb.shape[:2]
    identity = np.eye(3,dtype=np.float32)
    if min(height,width) < 20:
        return rgb,identity,False
    gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    _,mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours,key=cv2.contourArea,reverse=True):
        area = cv2.contourArea(contour)
        if area < .22*width*height:
            break
        rect = cv2.minAreaRect(contour)
        a,b = rect[1]
        if min(a,b) < 12 or max(a,b)/min(a,b) < 2.8 or area/(a*b) < .65:
            continue
        points = cv2.boxPoints(rect)
        sums,diffs = points.sum(axis=1),np.diff(points,axis=1).ravel()
        order = [sums.argmin(),diffs.argmin(),sums.argmax(),diffs.argmax()]
        if len(set(order)) != 4:
            continue
        src = points[order].astype(np.float32)
        if (src[:,0].min() < -2 or src[:,1].min() < -2 or
                src[:,0].max() > width+1 or src[:,1].max() > height+1):
            continue
        out_w = round(max(np.linalg.norm(src[1]-src[0]),np.linalg.norm(src[2]-src[3])))
        out_h = round(max(np.linalg.norm(src[3]-src[0]),np.linalg.norm(src[2]-src[1])))
        if out_w/max(out_h,1) < 2.8:
            continue
        dst = np.array([[0,0],[out_w-1,0],[out_w-1,out_h-1],[0,out_h-1]],dtype=np.float32)
        transform = cv2.getPerspectiveTransform(src,dst)
        return cv2.warpPerspective(rgb,transform,(out_w,out_h),flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE),transform,True
    return rgb,identity,False


def propose_first_pair(image, *,minimum_height_ratio=.18,maximum_overlap_ratio=.35):
    if (not math.isfinite(minimum_height_ratio) or not 0 < minimum_height_ratio < .9
            or not math.isfinite(maximum_overlap_ratio) or not 0 <= maximum_overlap_ratio <= .5):
        raise ValueError('invalid first-pair geometry parameters')
    plate,transform,rectified = plate_view(image)
    height,width = plate.shape[:2]
    if min(height,width) < 20:
        return None
    gray = cv2.cvtColor(plate,cv2.COLOR_RGB2GRAY)
    _,mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    inset = max(2,round(height*.04))
    mask[:inset,:] = 0
    mask[-inset:,:] = 0
    mask[:,:inset] = 0
    mask[:,-inset:] = 0
    count,_,stats,_ = cv2.connectedComponentsWithStats(mask,connectivity=8)
    components = []
    for x,y,w,h,area in stats[1:]:
        if (h < minimum_height_ratio*height or h > .9*height or w/h < .12 or w/h > 1.6
                or w > .28*width or area < .003*height*width
                or x+w > .66*width or y < inset or y+h >= height-inset):
            continue
        components.append([int(x),int(y),int(x+w),int(y+h)])
    components.sort(key=lambda box:box[0])
    for first,second in zip(components,components[1:]):
        h1,h2 = first[3]-first[1],second[3]-second[1]
        gap = second[0]-first[2]
        allowed_overlap = maximum_overlap_ratio*min(first[2]-first[0],second[2]-second[0])
        if (not .65 <= h1/h2 <= 1.55 or gap < -allowed_overlap or gap > .15*width
                or abs((first[1]+first[3])-(second[1]+second[3])) > .4*height
                or first[0] > .4*width):
            continue
        pad = max(2,round(max(h1,h2)*.06))
        box = [max(0,first[0]-pad),max(0,min(first[1],second[1])-pad),
               min(width,second[2]+pad),min(height,max(first[3],second[3])+pad)]
        points = np.array([[box[0],box[1]],[box[2]-1,box[1]],
                           [box[2]-1,box[3]-1],[box[0],box[3]-1]],dtype=np.float32)[None]
        source = cv2.perspectiveTransform(points,np.linalg.inv(transform))[0]
        if (source[:,0].min() < -1 or source[:,1].min() < -1 or
                source[:,0].max() > image.width or source[:,1].max() > image.height):
            return None
        plate_image=Image.fromarray(plate)
        return PrefixProposal(plate_image.crop(tuple(box)),plate_image,box,
                              source.astype(float).tolist(),transform.astype(float).tolist(),
                              rectified,components)
    return None
