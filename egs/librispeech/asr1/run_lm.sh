#!/bin/bash
ngpu=1
backend=pytorch
stage=3

# decoding parameter
do_delta=false # true when using CNN
beam_size=20
penalty=0.0
maxlenratio=0.8
minlenratio=0.3
ctc_weight=0.0
lm_weight=0.3
recog_model=acc.best # set a model to be used for decoding: 'acc.best' or 'loss.best'
nj=32

. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

train_set=train_100
train_dev=dev
recog_set="test_clean test_other dev_clean dev_other"

dict=data/lang_1char/${train_set}_units.txt
lmexpdir=exp/train_rnnlm_2layer_bs256
# mkdir -p ${lmexpdir}
# if [ ${stage} -le 3 ]; then
#     echo "stage 3: LM Preparation"
#     lmdatadir=data/local/lm_train
#     mkdir -p ${lmdatadir}
#     text2token.py -s 1 -n 1 data/${train_set}/text | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
#         > ${lmdatadir}/train.txt
#     text2token.py -s 1 -n 1 data/${train_dev}/text | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
#         > ${lmdatadir}/valid.txt
#     # use only 1 gpu
#     if [ ${ngpu} -gt 1 ]; then
#         echo "LM training does not support multi-gpu. signle gpu will be used."
#     fi
#     ${cuda_cmd} ${lmexpdir}/train.log \
#         lm_train.py \
#         --ngpu ${ngpu} \
#         --backend ${backend} \
#         --verbose 1 \
#         --outdir ${lmexpdir} \
#         --train-label ${lmdatadir}/train.txt \
#         --valid-label ${lmdatadir}/valid.txt \
#         --epoch 60 \
#         --batchsize 256 \
#         --dict ${dict}
# fi

config=./exp/train_100_blstmp_e8_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.0_adadelta_bs50_mli800_mlo150/results/model.conf
expdir=./exp/train_100_blstmp_e8_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.0_adadelta_bs50_mli800_mlo150
if [ ${stage} -le 4 ]; then
    echo "stage 4: Decoding"
    nj=32

    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}

        # split data
        data=data/${rtask}
        split_data.sh --per-utt ${data} ${nj};
        sdata=${data}/split${nj}utt;

        # feature extraction
        feats="ark,s,cs:apply-cmvn --norm-vars=true data/${train_set}/cmvn.ark scp:${sdata}/JOB/feats.scp ark:- |"
        if ${do_delta}; then
        feats="$feats add-deltas ark:- ark:- |"
        fi

        # make json labels for recognition
        data2json.sh ${data} ${dict} > ${data}/data.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-feat "$feats" \
            --recog-label ${data}/data.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/model.${recog_model}  \
            --model-conf ${config} \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} \
            --rnnlm ${lmexpdir}/rnnlm.model.best \
            --lm-weight ${lm_weight} \
            &
        wait

        score_sclite.sh --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    done
    wait
    echo "Finished"
fi

