"""Download reviewed public image URLs into a new, local review quarantine.

No dataset annotations, splits, model weights or raw inputs are changed.
License metadata is provenance, not a legal conclusion or label verification.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
HOSTS = {'upload.wikimedia.org', 'source.roboflow.com'}
LICENSES = {'CC-BY-3.0','CC-BY-4.0','CC-BY-SA-3.0','CC-BY-SA-4.0','CC0-1.0'}
MAX_BYTES = 12 * 1024 * 1024

def validate_asset_url(url):
    parts=urlsplit(url)
    if parts.scheme!='https' or parts.hostname not in HOSTS or parts.username or parts.password or parts.port not in (None,443):
        raise ValueError('asset URL must use HTTPS on an explicitly supported public media host')
    return url

class PublicRedirects(HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        validate_asset_url(newurl)
        return super().redirect_request(req,fp,code,msg,headers,newurl)

def inspect_bytes(payload, excluded_hashes):
    digest=hashlib.sha256(payload).hexdigest()
    # Compare before any image decoder sees potentially protected pixels.
    if digest in excluded_hashes:
        return {'sha256':digest,'status':'excluded_existing_hash'}
    if not payload or len(payload)>MAX_BYTES:
        raise ValueError('empty or oversized image')
    with Image.open(io.BytesIO(payload)) as image:
        width,height=image.size
        image_format=image.format
        if width*height>20_000_000 or image_format not in ('JPEG','PNG','WEBP'):
            raise ValueError('unsupported image format or pixel count')
        image.verify()
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        pixel_hash=hashlib.sha256(f'RGB:{width}x{height}:'.encode()+image.convert('RGB').tobytes()).hexdigest()
    return {'sha256':digest,'pixel_sha256':pixel_hash,'width':width,'height':height,
            'format':image_format,'status':'needs_visual_review',
            'small_image':width<160 or height<32}

def validate_entry(row):
    if not re.fullmatch(r'[a-z0-9_-]{1,64}',row.get('id','')):
        raise ValueError('invalid source id')
    if row.get('license_spdx') not in LICENSES:
        raise ValueError('source license must have been explicitly reviewed')
    for key in ('author','title','source_url','license_url'):
        if not isinstance(row.get(key),str) or not row[key].strip():
            raise ValueError('missing attribution field: '+key)
    for key in ('source_url','license_url'):
        if urlsplit(row[key]).scheme!='https':
            raise ValueError('provenance links must use HTTPS')
    validate_asset_url(row['asset_url'])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--protected-hashes-csv',type=Path,required=True,
                        help='existing filename,sha256 inventory; hashes only, no image decoding')
    args=parser.parse_args()
    output=args.output.resolve()
    boundary=(ROOT/'data/processed/external').resolve()
    if not output.is_relative_to(boundary) or output==boundary:
        raise ValueError('output must be a new subdirectory of data/processed/external')
    if output.exists():
        raise FileExistsError(output)
    rows=json.loads(args.manifest.read_text(encoding='utf-8'))['sources']
    if not rows or len({r['id'] for r in rows})!=len(rows):
        raise ValueError('empty or duplicate source inventory')
    for row in rows:
        validate_entry(row)
    with args.protected_hashes_csv.open(encoding='utf-8-sig',newline='') as stream:
        excluded={r['sha256'] for r in csv.DictReader(stream)}
    if any(not re.fullmatch('[0-9a-f]{64}',h) for h in excluded):
        raise ValueError('invalid exclusion hash inventory')
    output.mkdir(parents=True)
    opener=build_opener(PublicRedirects())
    results=[]
    seen_bytes=set()
    seen_pixels=set()
    for row in rows:
        record={**row,'training_admitted':False,'label':None}
        try:
            request=Request(row['asset_url'],headers={'User-Agent':'ANPR-System-public-data-review/1.0'})
            with opener.open(request,timeout=30) as response:
                validate_asset_url(response.geturl())
                payload=response.read(MAX_BYTES+1)
            info=inspect_bytes(payload,excluded)
            record.update(info)
            if info['status']!='excluded_existing_hash':
                if info['sha256'] in seen_bytes or info['pixel_sha256'] in seen_pixels:
                    record['status']='duplicate_in_download'
                else:
                    suffix={'JPEG':'.jpg','PNG':'.png','WEBP':'.webp'}[info['format']]
                    path=output/(row['id']+suffix)
                    path.write_bytes(payload)
                    record['local_file']=path.name
                    seen_bytes.add(info['sha256'])
                    seen_pixels.add(info['pixel_sha256'])
        except Exception as exc:
            record.update(status='download_or_integrity_failure',error_type=type(exc).__name__)
        results.append(record)
    report={'downloaded_at_utc':datetime.now(timezone.utc).isoformat(),
            'sources':results,'training_admitted':False,'raw_files_modified':False,
            'protected_pixels_decoded':False,'warning':'Quarantine only. Review text, country, license and split identity before admission.'}
    (output/'intake.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'sources':len(results),'saved':sum('local_file' in r for r in results),
                      'failed':sum(r['status']=='download_or_integrity_failure' for r in results),
                      'training_admitted':False}))
    return int(any(r['status']=='download_or_integrity_failure' for r in results))

if __name__=='__main__':
    raise SystemExit(main())
