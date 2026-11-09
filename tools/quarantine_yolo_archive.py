"""Audit a local public YOLO ZIP; extract only train/validation into quarantine.

Test members stay sealed. Upstream annotations remain unverified; this utility
never creates training datasets or evaluates recognition accuracy.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import re
import stat
import zipfile
from collections import Counter, defaultdict
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
import yaml
from tools.quarantine_public_images import ROOT,LICENSES,MAX_BYTES,inspect_bytes

IMAGE_SUFFIXES={'.jpg','.jpeg','.png','.webp'}

def safe_members(archive):
    infos=archive.infolist()
    if len(infos)>20000 or sum(i.file_size for i in infos)>2*1024**3:
        raise ValueError('archive exceeds conservative inventory/size bounds')
    seen={}
    unique=[]
    for info in infos:
        path=PurePosixPath(info.filename)
        if (path.is_absolute() or '..' in path.parts or chr(92) in info.orig_filename
                or ':' in info.filename or stat.S_ISLNK(info.external_attr>>16)):
            raise ValueError('unsafe archive member')
        if info.file_size>MAX_BYTES:
            raise ValueError('oversized archive member')
        canonical=str(path).casefold()
        if canonical in seen:
            # Some upstream exports repeat identical root YAML metadata.
            # Never resolve ambiguous duplicate image/annotation paths.
            if (info.filename!='data.yaml' or seen[canonical].filename!='data.yaml'
                    or info.file_size>65536 or archive.read(info)!=archive.read(seen[canonical])):
                raise ValueError('ambiguous duplicate archive member')
            continue
        seen[canonical]=info
        unique.append(info)
    return unique

def parse_annotations(text,class_count):
    boxes=[]
    geometries=[]
    for line in text.splitlines():
        if not line.strip():
            continue
        values=line.split()
        if len(values)!=5 and (len(values)<7 or len(values)%2!=1):
            raise ValueError('expected YOLO box or polygon coordinates')
        numbers=list(map(float,values))
        if not all(math.isfinite(x) for x in numbers):
            raise ValueError('non-finite annotation')
        class_value=numbers[0]
        if class_value!=int(class_value) or not 0<=class_value<class_count:
            raise ValueError('unknown annotation class')
        if len(numbers)==5:
            _,cx,cy,width,height=numbers
            geometries.append('box')
        else:
            coords=numbers[1:]
            if min(coords)<0 or max(coords)>1:
                raise ValueError('polygon outside normalized image')
            xs,ys=coords[::2],coords[1::2]
            cx,cy=(min(xs)+max(xs))/2,(min(ys)+max(ys))/2
            width,height=max(xs)-min(xs),max(ys)-min(ys)
            geometries.append('polygon_envelope_for_review')
        if (not 0<width<=1 or not 0<height<=1 or min(cx-width/2,cy-height/2)<-1e-6
                or max(cx+width/2,cy+height/2)>1+1e-6):
            raise ValueError('box outside normalized image')
        boxes.append([int(class_value),cx,cy,width,height])
    return boxes,geometries

def parse_boxes(text,class_count):
    return parse_annotations(text,class_count)[0]

def source_group(path):
    # Roboflow names retain the original source stem before .rf.<hash>.
    # This is a grouping hint only, not a plate identity or OCR annotation.
    return PurePosixPath(path).stem.split('.rf.',1)[0]

def audit_archive(zip_path,output,excluded_hashes,provenance):
    if output.exists():
        raise FileExistsError(output)
    with zipfile.ZipFile(zip_path) as archive:
        infos=safe_members(archive)
        by_name={i.filename:i for i in infos if not i.is_dir()}
        if 'data.yaml' not in by_name or by_name['data.yaml'].file_size>65536:
            raise ValueError('missing or oversized root data.yaml')
        metadata=yaml.safe_load(archive.read('data.yaml'))
        names=metadata.get('names') if isinstance(metadata,dict) else None
        if isinstance(names,dict):
            if sorted(names)!=list(range(len(names))):
                raise ValueError('non-contiguous class mapping')
            names=[names[i] for i in range(len(names))]
        if not isinstance(names,list) or not names or any(not isinstance(n,str) for n in names):
            raise ValueError('missing class names')
        if int(metadata.get('nc',len(names)))!=len(names):
            raise ValueError('class count differs from mapping')
        output.mkdir(parents=True)
        rows=[]
        sealed=[]
        split_counts=Counter()
        groups=defaultdict(set)
        for info in infos:
            member=PurePosixPath(info.filename)
            if info.is_dir():
                continue
            parts=member.parts
            if any(p.lower() in ('test','testing','holdout') for p in parts):
                sealed.append(info.filename)
                if member.suffix.lower() in IMAGE_SUFFIXES:
                    split_counts['test_sealed']+=1
                    groups[source_group(info.filename)].add('test_sealed')
                continue  # No read/decode/extraction of held-out member bytes.
            if (len(parts)!=3 or parts[0] not in ('train','valid','val')
                    or parts[1]!='images' or member.suffix.lower() not in IMAGE_SUFFIXES):
                continue
            split='val' if parts[0] in ('valid','val') else 'train'
            group=source_group(info.filename)
            groups[group].add(split)
            split_counts[split]+=1
            row={'member':info.filename,'split':split,'source_group_hint':group,
                 'training_admitted':False}
            payload=archive.read(info)
            try:
                row.update(inspect_bytes(payload,excluded_hashes))
            except Exception as exc:
                row.update(status='image_integrity_failure',error_type=type(exc).__name__)
            if row['status']=='needs_visual_review':
                destination=output/str(member)
                destination.parent.mkdir(parents=True,exist_ok=True)
                destination.write_bytes(payload)
                label_path=PurePosixPath(parts[0],'labels',member.stem+'.txt')
                if str(label_path) not in by_name:
                    row['annotation_status']='missing_not_background'
                else:
                    text=archive.read(str(label_path)).decode('utf-8-sig')
                    label_destination=output/str(label_path)
                    label_destination.parent.mkdir(parents=True,exist_ok=True)
                    label_destination.write_text(text,encoding='utf-8')
                    try:
                        row['boxes'],row['annotation_geometry']=parse_annotations(text,len(names))
                        row['annotation_status']='structurally_valid_unverified'
                    except ValueError as exc:
                        row['annotation_status']='invalid'
                        row['annotation_error']=str(exc)
            rows.append(row)
        cross_split=[{'source_group_hint':g,'splits':sorted(s)} for g,s in groups.items() if len(s)>1]
        hashes=defaultdict(list)
        pixels=defaultdict(list)
        for row in rows:
            if row.get('status')=='needs_visual_review':
                hashes[row['sha256']].append(row['member'])
                pixels[row['pixel_sha256']].append(row['member'])
        report={'source':provenance,'archive_sha256':hashlib.sha256(zip_path.read_bytes()).hexdigest(),
                'identical_metadata_entries_skipped':len(archive.infolist())-len(infos),
                'audited_at_utc':datetime.now(timezone.utc).isoformat(),'class_names':names,
                'split_image_counts':dict(split_counts),'source_group_hints':len(groups),
                'cross_split_source_group_hints':cross_split,
                'duplicate_byte_groups':[v for v in hashes.values() if len(v)>1],
                'duplicate_pixel_groups':[v for v in pixels.values() if len(v)>1],
                'image_status_counts':dict(Counter(r['status'] for r in rows)),
                'annotation_status_counts':dict(Counter(r.get('annotation_status','not_inspected') for r in rows)),
                'rows':rows,'sealed_test_members':sealed,'test_pixels_decoded':False,
                'training_admitted':False,'raw_files_modified':False,
                'limitations':'Filename groups and exact hashes do not establish plate-identity or near-duplicate independence. Source labels/country require visual review.'}
        (output/'archive_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--protected-hashes-csv',type=Path,required=True)
    parser.add_argument('--source-url',required=True)
    parser.add_argument('--author',required=True)
    parser.add_argument('--license-spdx',choices=sorted(LICENSES),required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    boundary=(ROOT/'data/processed/external').resolve()
    if not output.is_relative_to(boundary) or output==boundary:
        raise ValueError('output must be a new child of data/processed/external')
    with args.protected_hashes_csv.open(encoding='utf-8-sig',newline='') as stream:
        excluded={r['sha256'] for r in csv.DictReader(stream)}
    if any(not re.fullmatch('[0-9a-f]{64}',value) for value in excluded):
        raise ValueError('invalid protected hash inventory')
    report=audit_archive(args.archive.resolve(),output,excluded,
                         {'url':args.source_url,'author':args.author,'license_spdx':args.license_spdx})
    print(json.dumps({key:report[key] for key in ('split_image_counts','source_group_hints',
                     'image_status_counts','annotation_status_counts','training_admitted')},indent=2))

if __name__=='__main__':
    main()
