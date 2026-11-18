"""ZIP safety and sealed-test policy, using synthetic images and annotations."""
import hashlib
import io
import stat
import zipfile
import pytest
from PIL import Image
from tools.quarantine_yolo_archive import audit_archive,parse_boxes,safe_members,parse_annotations

def png():
    buffer=io.BytesIO()
    Image.new('RGB',(160,50),'white').save(buffer,format='PNG')
    return buffer.getvalue()

def test_test_members_are_never_read_or_extracted(tmp_path,monkeypatch):
    path=tmp_path/'source.zip'
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('data.yaml',"nc: 1\nnames: ['plate']\n")
        archive.writestr('train/images/example.rf.a.png',png())
        archive.writestr('train/labels/example.rf.a.txt','0 .5 .5 .5 .5')
        archive.writestr('test/images/sealed.png',b'must not read')
        archive.writestr('test/labels/sealed.txt',b'must not read')
    original=zipfile.ZipFile.read
    def guarded(self,name,*args,**kwargs):
        member=name.filename if isinstance(name,zipfile.ZipInfo) else name
        assert not member.startswith('test/')
        return original(self,name,*args,**kwargs)
    monkeypatch.setattr(zipfile.ZipFile,'read',guarded)
    report=audit_archive(path,tmp_path/'out',set(),{'author':'synthetic'})
    assert report['split_image_counts']=={'test_sealed':1,'train':1}
    assert not (tmp_path/'out/test').exists()
    assert report['training_admitted'] is False

@pytest.mark.parametrize('member',['../outside.txt','/absolute.txt','train\\images\\bad.png','C:/outside.txt'])
def test_rejects_unsafe_members_before_extraction(tmp_path,member):
    path=tmp_path/'unsafe.zip'
    with zipfile.ZipFile(path,'w') as archive:
        info=zipfile.ZipInfo('placeholder')
        # ZipInfo normalizes separators on Windows; deliberately preserve the
        # raw hostile archive name to verify behavior on both host platforms.
        info.filename=member
        archive.writestr(info,'bad')
    with zipfile.ZipFile(path) as archive,pytest.raises(ValueError):
        safe_members(archive)

@pytest.mark.parametrize('label',['0 nan .5 .5 .5','0 .9 .5 .5 .5','2 .5 .5 .1 .1','0 .5 .5 -1 .1'])
def test_rejects_invalid_boxes(label):
    with pytest.raises(ValueError):
        parse_boxes(label,1)

def test_existing_hash_skips_image_decode_and_copy(tmp_path,monkeypatch):
    payload=b'protected synthetic bytes'
    path=tmp_path/'source.zip'
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('data.yaml',"names: ['plate']\n")
        archive.writestr('train/images/protected.png',payload)
    monkeypatch.setattr(Image,'open',lambda *a,**k: pytest.fail('protected pixels decoded'))
    report=audit_archive(path,tmp_path/'out',{hashlib.sha256(payload).hexdigest()},{})
    assert report['rows'][0]['status']=='excluded_existing_hash'
    assert not (tmp_path/'out/train').exists()

def test_polygon_envelope_retains_source_geometry_provenance():
    boxes,kinds=parse_annotations('0 .2 .3 .8 .3 .7 .9 .3 .8',1)
    assert kinds==['polygon_envelope_for_review']
    assert boxes[0]==pytest.approx([0,.5,.6,.6,.6])

def test_only_identical_root_metadata_duplicates_are_accepted(tmp_path):
    path=tmp_path/'metadata.zip'
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('data.yaml','same')
        with pytest.warns(UserWarning):
            archive.writestr('data.yaml','same')
    with zipfile.ZipFile(path) as archive:
        assert len(safe_members(archive))==1
    with zipfile.ZipFile(path,'a') as archive,pytest.warns(UserWarning):
        archive.writestr('data.yaml','different')
    with zipfile.ZipFile(path) as archive,pytest.raises(ValueError):
        safe_members(archive)

@pytest.mark.parametrize('alias',['train/images/EXAMPLE.png','train//images/example.png'])
def test_rejects_case_and_separator_aliases_on_every_platform(tmp_path,alias):
    path=tmp_path/'alias.zip'
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('train/images/example.png','first')
        archive.writestr(alias,'different')
    with zipfile.ZipFile(path) as archive,pytest.raises(ValueError):
        safe_members(archive)

def test_symlink_member_is_rejected(tmp_path):
    path=tmp_path/'link.zip'
    with zipfile.ZipFile(path,'w') as archive:
        info=zipfile.ZipInfo('train/images/link.png')
        info.create_system=3
        info.external_attr=(stat.S_IFLNK|0o777)<<16
        archive.writestr(info,'/outside')
    with zipfile.ZipFile(path) as archive,pytest.raises(ValueError):
        safe_members(archive)
