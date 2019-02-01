#!/bin/bash -e

nj=4
cmd=run.pl
write_utt2num_frames=true
compress=true
filetype=mat # mat or hdf5

. utils/parse_options.sh || exit 1

set -euo pipefail

if [ ! $# -eq 2 ]; then
   echo "Usage: $0 [options] <data-dir> <bert-dir> ";
   echo "e.g.: $0 data/train exp/bert"
   echo "Options: "
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
bertdir=$2

# check bert installation
script=../../../tools/bert/extract_features.py
if [ ! -e ${script} ]; then
    echo "It seems that bert is not installed."
    echo "Please try following commands:"
    echo "cd ../../../tools; make bert.done"
    exit 1
fi

if [ ! -e ${bertdir} ];then
    wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip
    unzip uncased_L-24_H-1024_A-16.zip
    mv uncased_L-24_H-1024_A-16 ${bertdir}
fi

name=$(basename ${data})
outdir=${bertdir}/${name}
[ ! -e ${outdir} ] && mkdir -p ${outdir}

# split text into the nj subsets
split_texts=""
for n in $(seq ${nj}); do
    split_texts="${split_texts} ${outdir}/text.${n}"
done
utils/split_scp.pl ${data}/text ${split_texts} || exit 1;
for n in $(seq ${nj}); do
    cat ${outdir}/text.${n} | cut -d " " -f 2- > ${outdir}/input.${n}.txt
done

# extract bert feature as json
${cmd} --gpu 1 JOB=1:${nj} ${outdir}/log/extract_bert_${name}.JOB.log \
     python ${script} \
        --input_file=${outdir}/input.JOB.txt \
        --output_file=${outdir}/output.JOB.json \
        --vocab_file=${bertdir}/vocab.txt \
        --bert_config_file=${bertdir}/bert_config.json \
        --init_checkpoint=${bertdir}/bert_model.ckpt \
        --layers=-1 \
        --max_seq_length=128 \
        --batch_size=8

# convert json to ark and scp
if ${write_utt2num_frames}; then
  write_num_frames_opt="--write-num-frames=ark,t:${outdir}/utt2num_frames.JOB"
else
  write_num_frames_opt=
fi

${cmd} --gpu 0 JOB=1:${nj} ${outdir}/log/convert_from_json_${name}.JOB.log \
    python local/convert_bert_feature.py \
        ${write_num_frames_opt} \
        --compress=${compress} \
        --filetype ${filetype} \
        ${outdir}/text.JOB \
        ${outdir}/output.JOB.json \
        ark,scp:${outdir}/bert_feats.JOB.ark,${outdir}/bert_feats.JOB.scp

for n in $(seq $nj); do
    cat ${outdir}/bert_feats.${n}.scp || exit 1;
done > ${outdir}/bert_feats.scp || exit 1

if ${write_utt2num_frames}; then
    for n in $(seq ${nj}); do
        cat ${outdir}/utt2num_frames.${n} || exit 1;
    done > ${outdir}/utt2num_frames || exit 1
    rm ${outdir}/utt2num_frames.* 2>/dev/null
fi
