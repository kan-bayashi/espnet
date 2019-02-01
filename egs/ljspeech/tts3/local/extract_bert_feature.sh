#!/bin/bash -e

nj=32
decode_sets="train_no_dev train_dev eval"
bert_base_dir=exp/bert_uncased_L-24_H-1024_A-16
script=/work4/t_hayashi/work/bert/extract_features.py

if [ ! -e ${bert_base_dir} ];then
    wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip
    unzip uncased_L-24_H-1024_A-16.zip
    mv uncased_L-24_H-1024_A-16 ${bert_base_dir}
fi

for decode_set in ${decode_sets};do
    outdir=${bert_base_dir}/${decode_set}
    [ ! -e ${outdir} ] && mkdir -p ${outdir}
    cat data/${decode_set}/text | awk '{print $2}' > ${outdir}/input.txt
    python ${script} \
        --input_file=${outdir}/input.txt \
        --output_file=${outdir}/output.json \
        --vocab_file=${bert_base_dir}/vocab.txt \
        --bert_config_file=${bert_base_dir}/bert_config.json \
        --init_checkpoint=${bert_base_dir}/bert_model.ckpt \
        --layers=-1 \
        --max_seq_length=128 \
        --batch_size=8

    python local/convert_bert_feature.py --nj ${nj} \
        data/${decode_set}/text ${outdir}/output.json ${outdir}/bert_vector

    for n in $(seq $nj); do
        cat ${outdir}/bert_vector.$n.scp || exit 1;
    done > ${outdir}/bert_vector.scp || exit 1
done
