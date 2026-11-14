"""Evaluate explicitly selected YOLO train/val images; never decode protected images.

Supply trusted local weights and a private dataset directory containing
train/images, train/labels, val/images and val/labels. Training diagnostics use
unique images, regardless of oversampling lists. Outputs contain private paths
and predictions and must stay outside version control.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from tools.detector_metrics import match_plate_boxes

SUFFIXES={".jpg",".jpeg",".png"}


def read_labels(path):
    rows=[]
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():continue
        fields=line.split()
        if len(fields)!=5 or fields[0]!="0":raise ValueError("expected single-class YOLO plate label")
        values=list(map(float,fields[1:]))
        if not all(math.isfinite(v) for v in values):raise ValueError("non-finite annotation")
        cx,cy,w,h=values
        if not 0<w<=1 or not 0<h<=1 or min(cx-w/2,cy-h/2)<-1e-6 or max(cx+w/2,cy+h/2)>1+1e-6:
            raise ValueError("annotation outside image or zero area")
        rows.append([max(0.,cx-w/2),max(0.,cy-h/2),min(1.,cx+w/2),min(1.,cy+h/2)])
    return rows


def preflight(data_dir,split,protected_images):
    if split not in ("train","val"):raise ValueError("only explicit development train/val splits are allowed")
    root=Path(data_dir).resolve();protected=Path(protected_images).resolve()
    if not protected.is_dir():raise ValueError("protected image inventory must exist")
    sealed=[p for p in protected.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES]
    if not sealed:raise ValueError("empty protected image inventory")
    protected_names={p.name for p in sealed};protected_hashes={hashlib.sha256(p.read_bytes()).hexdigest() for p in sealed}
    folder=root/split/"images"
    if not folder.is_dir():raise ValueError("development image directory missing")
    rows=[]
    for image in sorted(folder.iterdir()):
        if not image.is_file() or image.suffix.lower() not in SUFFIXES:continue
        if not image.resolve().is_relative_to(root):raise ValueError("source escapes data directory")
        digest=hashlib.sha256(image.read_bytes()).hexdigest()
        if image.name in protected_names or digest in protected_hashes:raise ValueError("protected image rejected before decoding")
        label=root/split/"labels"/(image.stem+".txt")
        if not label.is_file() or not label.resolve().is_relative_to(root):raise ValueError("missing or unsafe annotation; backgrounds require an explicit empty label")
        rows.append({"image":image,"label":label,"sha256":digest,"label_sha256":hashlib.sha256(label.read_bytes()).hexdigest(),"normalized_boxes":read_labels(label)})
    if not rows:raise ValueError("empty development split")
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--split",choices=("train","val"),default="val")
    parser.add_argument("--protected-images",type=Path,required=True)
    parser.add_argument("--weights",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--confidence",type=float,default=.25)
    parser.add_argument("--imgsz",type=int,default=1280)
    args=parser.parse_args()
    if args.imgsz<32 or args.imgsz%32:raise ValueError("image size must be a positive multiple of 32")
    if not math.isfinite(args.confidence) or not 0<args.confidence<=1:raise ValueError("invalid confidence threshold")
    if args.output.exists():raise FileExistsError(args.output)
    samples=preflight(args.data_dir,args.split,args.protected_images)
    if not args.weights.is_file():raise FileNotFoundError(args.weights)
    weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest()
    from ultralytics import YOLO
    model=YOLO(str(args.weights))
    if len(model.names)!=1:raise ValueError("expected a single-class plate detector")
    rows=[]
    for sample in samples:
        # Recheck bytes before the model's decoder to catch source replacement.
        if hashlib.sha256(sample["image"].read_bytes()).hexdigest()!=sample["sha256"]:raise ValueError("source changed after preflight")
        if hashlib.sha256(sample["label"].read_bytes()).hexdigest()!=sample["label_sha256"]:raise ValueError("annotation changed after preflight")
        result=model.predict(str(sample["image"]),imgsz=args.imgsz,conf=args.confidence,iou=.5,max_det=300,device=args.device,verbose=False)[0]
        height,width=result.orig_shape
        truth=[[a*width,b*height,c*width,d*height] for a,b,c,d in sample["normalized_boxes"]]
        if any(int(c)!=0 for c in result.boxes.cls.cpu().tolist()):raise ValueError("detector is not a single-class plate model")
        boxes=result.boxes.xyxy.cpu().tolist();scores=result.boxes.conf.cpu().tolist()
        ordered=sorted(zip(scores,boxes),key=lambda item:item[0],reverse=True)
        boxes=[b for _,b in ordered];scores=[s for s,_ in ordered]
        rows.append({"image":str(sample["image"]),"sha256":sample["sha256"],"label_sha256":sample["label_sha256"],"truth":truth,"predicted_boxes":boxes,"scores":scores,
                     **match_plate_boxes(boxes,truth)})
    summary={k:sum(r[k] for r in rows) for k in ("tp","fp","fn","duplicate_boxes","other_false_boxes")}
    summary["precision"]=summary["tp"]/(summary["tp"]+summary["fp"]) if summary["tp"]+summary["fp"] else None
    summary["recall"]=summary["tp"]/(summary["tp"]+summary["fn"]) if summary["tp"]+summary["fn"] else None
    if hashlib.sha256(args.weights.read_bytes()).hexdigest()!=weights_sha256:raise ValueError("checkpoint changed during evaluation")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({"summary":summary,"rows":rows,"split":args.split,"weights_sha256":weights_sha256,"settings":{"imgsz":args.imgsz,"conf":args.confidence,"nms_iou":.5,"matching_iou":.5,"max_det":300,"application_plate_cap_applied":False},"protected_pixels_decoded":False,"limitation":"Reused development data, not independent field accuracy; train reports are fit diagnostics. Positive-only cohorts cannot estimate background false-positive rate. YOLO NMS and its maximum remain active."},indent=2),encoding="utf-8")
    print(json.dumps(summary),flush=True)

if __name__=="__main__":main()
