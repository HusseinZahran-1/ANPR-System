"""Optional visual authority reading; never infer country or rewrite characters."""
from __future__ import annotations
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import numpy as np
from app.ocr.regions import propose_first_pair

ENGLISH_G2_SHA256='e2272681d9d67a04e2dff396b6e95077bc19001f8f6d3593c307b9852e1c29e8'


class AuthorityReaderUnavailable(RuntimeError):
    """Configured optional model/dependency cannot be used."""


@lru_cache(maxsize=2)
def load_prefix_reader(model_path: str, device: str):
    path=Path(model_path)
    if not path.is_file():
        raise FileNotFoundError('configured authority OCR model is missing')
    if path.name != 'english_g2.pth' or hashlib.sha256(path.read_bytes()).hexdigest()!=ENGLISH_G2_SHA256:
        raise AuthorityReaderUnavailable('authority OCR weights failed verification')
    try:
        import easyocr
    except ImportError as exc:
        raise AuthorityReaderUnavailable('optional authority OCR dependency is not installed') from exc
    reader=easyocr.Reader(['en'],gpu=device if device.startswith('cuda') else False,
                          detector=False,download_enabled=False,quantize=False,
                          model_storage_directory=str(path.parent),verbose=False)
    def predict(image):
        result=reader.recognize(np.asarray(image.convert('L')),detail=1,workers=0,paragraph=False,
                                allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
        return [{'text':p[1],'confidence':float(p[2])} for p in result]
    return predict


class AuthorityRegionReader:
    """Compose independently recognized prefix and spatial numeric suffix.

    No expected plate transcript is accepted as input. Numeric and other
    prefixes decline the proposal. Caller retains the primary reading whenever
    evidence is missing, ambiguous, below threshold or not purely numeric.
    """
    def __init__(self,prefix_reader,token_reader,*,min_prefix_confidence=.8):
        if not math.isfinite(min_prefix_confidence) or not 0 < min_prefix_confidence <= 1:
            raise ValueError('invalid authority prefix confidence threshold')
        self.prefix_reader=prefix_reader
        self.token_reader=token_reader
        self.min_prefix_confidence=min_prefix_confidence

    def read(self,image):
        proposal=propose_first_pair(image)
        if proposal is None:
            return None
        readings=self.prefix_reader(proposal.crop)
        if len(readings)!=1:
            return None
        prefix=readings[0]['text'].upper()
        confidence=float(readings[0]['confidence'])
        if (prefix not in ('CD','SN') or not math.isfinite(confidence)
                or not self.min_prefix_confidence <= confidence <= 1):
            return None
        tokens=self.token_reader(np.asarray(proposal.plate))
        suffix_tokens=[t for t in tokens if t['source_x_center'] >= proposal.box_xyxy[2]]
        suffix=''.join(t['character'] for t in suffix_tokens)
        if not suffix.isascii() or not suffix.isdigit() or not 1 <= len(suffix) <= 6:
            return None
        return {'text':prefix+suffix,'prefix_text':prefix,'numeric_suffix':suffix,
                'prefix_confidence':confidence,
                'suffix_confidence':sum(t['confidence'] for t in suffix_tokens)/len(suffix_tokens),
                'confidence_calibrated':False,'country_verified':False,'requires_human_review':True,
                'coordinate_space':'plate_crop','prefix_polygon':proposal.source_polygon,
                'source_to_plate':proposal.source_to_plate,'suffix_tokens':suffix_tokens,
                'method':'visual_prefix_and_spatial_ctc_suffix'}
