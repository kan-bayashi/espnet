#!/usr/bin/env python3

import argparse
import re
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="path to audiocaps wav data directory")
    parser.add_argument("annotations_dir", type=str, help="path to audiocaps annotation data directory")
    parser.add_argument("out_dirname", type=str, help="name of output data directory")

    args = parser.parse_args()
    
    data_dir = args.data_dir
    file_names = os.path.join(args.annotations_dir, 'filenames.txt')
    file_captions = os.path.join(args.annotations_dir, 'captions.txt')
    out_dir = os.path.join('data',args.out_dirname)
    
    filepaths_dict = {}
    with open(file_names,'r') as f:
        for line in f.readlines():
            file_id = 'file_audiocaps_'+line[:-1].split(' ')[0].split('_')[1].zfill(6)
            filename = line[:-1].split(' ')[1]
            filepath = os.path.join(data_dir, filename)
            if os.path.isfile(filepath):
                filepaths_dict[file_id] = filepath

    filecaptions_dict = {}
    with open(file_captions,'r') as f:
        for line in f.readlines():
            file_id = 'file_audiocaps_'+line[:-1].split(' ')[0].split('_')[1].zfill(6) # line[:-1].split(' ')[0]
            caption = ' '.join(line[:-1].split(' ')[1:])
            if file_id in filepaths_dict:
                filecaptions_dict[file_id] = caption

    filespeakers_dict = {key:'spk_audiocaps_'+f"{key.split('_')[-1]}".zfill(6) for key in filecaptions_dict}
    # filespeakers_dict = {key:f"spk_{key}" for key in filecaptions_dict}

    if not os.path.isdir(out_dir): os.makedirs(out_dir)

    with open(os.path.join(out_dir,'text'), 'w') as f:
        for key,val in filecaptions_dict.items():
            f.write(f'{key} {val}\n')

    with open(os.path.join(out_dir,'utt2spk'), 'w') as f:
        for key,val in filespeakers_dict.items():
            f.write(f'{key} {val}\n')
            
    with open(os.path.join(out_dir,'spk2utt'), 'w') as f:
        for key,val in filespeakers_dict.items():
            f.write(f'{val} {key}\n')
            
    with open(os.path.join(out_dir,'wav.scp'), 'w') as f:
        for key,val in filepaths_dict.items():
            f.write(f'{key} {val}\n')