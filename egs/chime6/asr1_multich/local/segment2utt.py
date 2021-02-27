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
    text = os.path.join(path,'text')
    
    with open(wav_scp,'r') as f: lines = f.readlines()
    wav_filedict = {}
    for line in lines:
        key,val = line[:-1].split(' ')
        wav_filedict[key] = sf.read(val)
    print('loaded required .wav files to memory')
    adir = os.path.dirname(val)
    sdir = os.path.dirname(os.path.abspath(adir))
    sdir = os.path.join(sdir,os.path.basename(os.path.abspath(path))+'_utt')
    if not os.path.isdir(sdir): os.mkdir(sdir)

    with open(segments,'r') as f: lines = f.readlines()
    print(f'preparing {len(lines)} utterances in {sdir}')
    with open(text,'r') as f: text_lines = f.readlines()
    assert len(lines) == len(text_lines), ValueError(f'unequal lengths of segments ({len(lines)}) and text ({len(text_lines)})')
    wav_new_lines = []; text_new_lines = []
    for line,text_line in zip(lines,text_lines):
        file, wav_idx, start_time, end_time = line[:-1].split(' ')
        start_time, end_time = float(start_time), float(end_time)

        ch = file.split('.')[1].split('-')[0]
        if ch[0] == 'C':
            utt_file = f"{file.split('.')[0]}-{int(100*start_time)}-{int(100*end_time)}.{ch}.wav"
        else:
            utt_file = f"{file.split('.')[0]}-{int(100*start_time)}-{int(100*end_time)}.wav"
        utt_filepath = os.path.join(sdir,utt_file)
        
        x,f = wav_filedict[wav_idx]
        # skip if required segment not in wav file
        if round(start_time*f) >= x.shape[0]: continue
        start_idx, end_idx = round(start_time*f), round(end_time*f)
        if not os.path.isfile(utt_filepath): sf.write(utt_filepath,,f)

        wav_line = ' '.join([file, utt_filepath+line[-1]])
        wav_new_lines.append(wav_line)
        text_new_lines.append(text_line)
    if len(lines) != len(text_new_lines):
        print(f'removed {len(lines)-len(text_new_lines)} segments due to zero wav-file length')
    with open(wav_scp,'w') as f: f.writelines(wav_new_lines)
    with open(text,'w') as f: f.writelines(text_new_lines)
    os.remove(segments)