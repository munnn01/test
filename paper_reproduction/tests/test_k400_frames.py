import csv
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from reproduce import data

def test_provider_validation_frames_join_official_labels_and_load_multiframe_clip(tmp_path):
    assert hasattr(data,'kinetics_frame_records'),'official-label frame validation adapter is not implemented'
    folder=tmp_path/'--7VUM9MKg4';folder.mkdir()
    for i in range(4):Image.new('RGB',(40,32),(i*25,0,0)).save(folder/f'img_{i+1:05}.jpg')
    annotation=tmp_path/'val.csv'
    annotation.write_text('label,youtube_id,time_start,time_end,split\nabseiling,--7VUM9MKg4,0,10,val\n')
    rows=data.kinetics_frame_records(tmp_path,annotation)
    assert len(rows)==1 and rows[0]['split']=='val' and rows[0]['id']=='kinetics/--7VUM9MKg4'
    x,label,meta=data.load_sample(rows[0],{'npp_size':32,'frames':2},False)
    assert x.shape==(1,2,3,32,32) and int(label)==0
    assert meta['source_format']=='provider_jpeg_frames' and meta['provider_cadence']=='unverified'

def test_official_got_sequence_list_does_not_stat_every_frame_directory(tmp_path):
    # Dataset mounts can expose a large manifest before per-file cold metadata is warm.
    # The loader, not the index, validates the selected source before use.
    (tmp_path/'list.txt').write_text('GOT-10k_Train_000001\nGOT-10k_Train_000002\n')
    rows=data.got_records(tmp_path,'train')
    assert len(rows)==2
    with pytest.raises((ValueError,OSError)):data.load_sample(rows[0],{'npp_size':32,'frames':2},False)
