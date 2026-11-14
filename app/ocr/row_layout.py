"""Conservative two-row geometry; no transcript, OCR score or country rules.

Proposals expose their cut and component evidence. Unfolding returns a piecewise
source-to-output mapping in pixel-edge coordinates; the source is never edited.
"""
from __future__ import annotations
import cv2
import math
import numpy as np
from PIL import Image


def propose_two_row_cut(image, *, minimum_x_ratio=.28, maximum_component_aspect=1.3):
    if (not math.isfinite(minimum_x_ratio) or not 0 <= minimum_x_ratio < .5
            or not math.isfinite(maximum_component_aspect) or not 1 <= maximum_component_aspect <= 3):
        raise ValueError("invalid row geometry limits")
    width,height=image.size
    if height<32 or not 1.8 <= width/height < 3.2:
        return None
    gray=np.asarray(image.convert("L"))
    _,mask=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    inset=max(2,round(height*.03))
    mask[:inset,:]=0; mask[-inset:,:]=0; mask[:,:inset]=0; mask[:,-inset:]=0
    _,_,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=8)
    components=[]
    for x,y,w,h,area in stats[1:]:
        if (x<minimum_x_ratio*width or h<.18*height or h>.62*height or not .12<=w/h<=maximum_component_aspect
                or area<.003*width*height or x+w>=width-inset):
            continue
        components.append([int(x),int(y),int(x+w),int(y+h)])
    if len(components)<5:
        return None
    # The split is derived from the largest vertical gap between glyph centers.
    ordered=sorted(components,key=lambda b:b[1]+b[3])
    gaps=[((ordered[i+1][1]+ordered[i+1][3]-ordered[i][1]-ordered[i][3])/2,i+1) for i in range(1,len(ordered)-2)]
    if not gaps: return None
    gap,split=max(gaps)
    top,bottom=ordered[:split],ordered[split:]
    if len(top)<2 or len(bottom)<3 or gap<.28*height:
        return None
    top_height=np.median([b[3]-b[1] for b in top]); bottom_height=np.median([b[3]-b[1] for b in bottom])
    if not .65<=top_height/bottom_height<=1.55:
        return None
    top_end=max(b[3] for b in top); bottom_start=min(b[1] for b in bottom)
    if bottom_start-top_end<.015*height:
        return None
    left=max(min(b[0] for b in top),min(b[0] for b in bottom)); right=min(max(b[2] for b in top),max(b[2] for b in bottom))
    min_span=min(max(b[2] for b in top)-min(b[0] for b in top),max(b[2] for b in bottom)-min(b[0] for b in bottom))
    if right-left<.3*min_span:
        return None
    return {"cut":round((top_end+bottom_start)/2),"components":components,"top":top,"bottom":bottom}



def unfold_at_cut(image: Image.Image, cut: int) -> tuple[Image.Image, dict]:
    """Place the complete upper image before the complete lower image.

    Mapping matrices use pixel edges: source [x0,y0,x1,y1] maps exactly to
    destination edges. OCR timestep centers are not represented as glyph boxes.
    """
    width,height=image.size
    if isinstance(cut,bool) or not isinstance(cut,int) or not 0<cut<height:
        raise ValueError("cut must be an integer strictly inside the image")
    target_height=max(cut,height-cut)
    rows=[(0,cut),(cut,height)]
    pieces=[];mapping=[];offset=0
    for y0,y1 in rows:
        target_width=round(width*target_height/(y1-y0))
        pieces.append(image.crop((0,y0,width,y1)).resize((target_width,target_height),Image.Resampling.BICUBIC))
        sx=target_width/width;sy=target_height/(y1-y0)
        mapping.append({"source_box_xyxy":[0,y0,width,y1],
                        "output_box_xyxy":[offset,0,offset+target_width,target_height],
                        "source_to_output":[[sx,0,offset],[0,sy,-sy*y0],[0,0,1]]})
        offset+=target_width
    result=Image.new(image.mode,(offset,target_height))
    offset=0
    for piece in pieces:
        result.paste(piece,(offset,0));offset+=piece.width
    return result,{"coordinate_convention":"pixel_edges","source_size":list(image.size),
                   "output_size":list(result.size),"cut":cut,"pieces":mapping}
