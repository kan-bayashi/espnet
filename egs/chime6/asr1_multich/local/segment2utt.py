#!/usr/bin/env python3

import os
import argparse
import soundfile as sf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("segments_dir", type=str, help="Path to segments and wav.scp directory")
    args = parser.parse_args()
    
    path = args.segments_dir
    
    segments = os.path.join(path,'segments')
    wav_scp = os.path.join(path,'wav.scp')
    
    with open(wav_scp,'r') as f: lines = f.readlines()
    wav_filedict = {}
    for line in lines:
        key,val = line[:-1].split(' ')
        wav_filedict[key] = sf.read(val)
    adir = os.path.dirname(val)
    sdir = os.path.dirname(os.path.abspath(adir))
    sdir = os.path.join(sdir,os.path.basename(os.path.abspath(path))+'_utt')
    if not os.path.isdir(sdir): os.mkdir(sdir)

    with open(segments,'r') as f: lines = f.readlines()
    print(f'preparing {len(lines)} utterances in {sdir}')
    for line in lines:
        file, wav_idx, start_time, end_time = line[:-1].split(' ')
        start_time,end_time = float(start_time),float(end_time)

        ch = file.split('.')[1].split('-')[0]
        if ch[0] == 'C':
            utt_file = f"{file.split('.')[0]}-{int(100*start_time)}-{int(100*end_time)}.{ch}.wav"
        else:
            utt_file = f"{file.split('.')[0]}-{int(100*start_time)}-{int(100*end_time)}.wav"
        utt_filepath = os.path.join(sdir,utt_file)
        
        x,f = wav_filedict[wav_idx]
        start_idx = round(start_time*f); end_idx = round(end_time*f)
        sf.write(utt_filepath,x[start_idx:end_idx],f)
        