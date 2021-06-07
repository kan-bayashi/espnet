#!/usr/bin/env bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=-1       # start from 0 if you need to start from data preparation
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

# preprocess_config=conf/no_preprocess.yaml
preprocess_config=conf/specaug.yaml
train_config=conf/train.yaml                   # conf/tuning/train_conformer_small.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml          # conf/decode.yaml

# rnnlm related
lm_vocabsize=65000
lm_resume=        # specify a snapshot file to resume LM training
lmtag=            # tag for managing LMs

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
n_average=10
use_valbest_average=false
use_custom_model=""

lang=en # en de fr cy tt kab ca zh-TW it fa eu es ru tr nl eo zh-CN rw pt zh-HK cs pl uk 

# exp tag
tag="" # tag for managing experiments.

# flags for clotho-v2 dataset
download_clothov2=false

# flags for audiocaps dataset
download_audiocaps=false
augment_audiocaps=true

# flag for performing speed perturbation augmentation
augment_speedperturbation=false

# flag for setting caption evaluation tools
download_evalmetrics=false

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=dev_clothov2
train_dev=val_clothov2
test_set=                                             # test_clothov2
recog_set="recog_val_clothov2 recog_eval_clothov2"
if ${augment_audiocaps} || ${download_audiocaps}; then
    train_set_without_audiocaps=${train_set}
    train_set=${train_set}_audiocaps
fi

clothov2_data_urlid="1-0s5SVN-SagFHDql95JAcfdn1GW4V7o9"
clothov2_kaldidata_urlid="1XE9ZKqW2sSMfA3xpdKl6ZyG2CFsIbbBQ"

audiocaps_train_urlid="1-QO73Elfdhtx8Jo918eJlksZKvTAoidx"
audiocaps_annotations_urlid="1Fn-dBl6SCKVukq1gMId4LTGBfrE3DJ-r"


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases 

    ### ---- below lines are for preparing clotho-v2 dataset ----
    if ${download_clothov2}; then
        echo "Downloading and Preparing Clotho-V2 Dataset"
        mkdir -p tmp
        wget "https://drive.google.com/uc?export=download&id=${clothov2_kaldidata_urlid}" -O "tmp/clothov2_kaldidata.tar.gz"
        tar -xzf "tmp/clothov2_kaldidata.tar.gz" -C ./
        ./local/download_large_drive_file.sh ${clothov2_data_urlid} "tmp/clothov2_data.tar.gz"
        tar -xzf "tmp/clothov2_data.tar.gz" -C ./
    fi

    ### ---- below lines are for preparing audiocaps dataset ----
    if ${download_audiocaps}; then
        echo "Downloading and Preparing AudioCaps Dataset"
        mkdir -p tmp
        wget "https://drive.google.com/uc?export=download&id=${audiocaps_annotations_urlid}" -O "tmp/annotations.tar.gz"
        tar -xzf "tmp/annotations.tar.gz" -C ./
        ./local/download_large_drive_file.sh ${audiocaps_train_urlid} "tmp/audiocaps_data_train.tar.gz"
        tar -xzf "tmp/audiocaps_data_train.tar.gz" -C ./
        python local/data_prep_audiocaps.py audiocaps_data/train audiocaps_data/annotations/train dev_audiocaps
        utils/combine_data.sh data/${train_set} data/${train_set_without_audiocaps} data/dev_audiocaps
    fi

    ### ---- below lines are for speed perturbation based augmentation ----
    if ${augment_speedperturbation}; then
        utils/perturb_data_dir_speed.sh 0.9 data/${train_set} data/temp1
        utils/perturb_data_dir_speed.sh 1.0 data/${train_set} data/temp2
        utils/perturb_data_dir_speed.sh 1.1 data/${train_set} data/temp3
        utils/combine_data.sh --extra-files utt2uniq data/${train_set} data/temp1 data/temp2 data/temp3
        rm -r data/temp1 data/temp2 data/temp3
    fi
    
    ### ---- below lines are for preparing caption evaluation tools ----
    if ${download_evalmetrics}; then
        echo "Download and Prepare Evaluation Metrics"
        mkdir -p tmp
        pip3 install scikit-image
        git clone https://github.com/audio-captioning/dcase-2020-baseline.git tmp/dcase_baseline
        cp -r tmp/dcase_baseline/coco_caption ./local/
        cp tmp/dcase_baseline/eval_metrics.py ./local/
        chmod -R 755 tmp/dcase_baseline
        rm -r tmp/dcase_baseline
        ./local/coco_caption/get_stanford_models.sh
    fi
    echo "Data Preparation Successful"
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank/${lang}
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in ${train_set} ${train_dev} ${recog_set}; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 16 --write_utt2num_frames true \
                                  data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data/${x}
    done

    # for x in ${train_set} ${train_dev} ${recog_set}; do
    #     # Remove features with too long frames in training data
    #     max_len=3000
    #     if [ -d "data/${x}/${x}_temp" ]; then
    #         rm -r data/${x}/${x}_temp
    #     fi
    #     remove_longshortdata.sh --maxframes $max_len data/${x} data/${x}_temp
    #     mv data/${x}_temp data/${x}
    # done

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    dump.sh --cmd "$train_cmd" --nj 16 --do_delta ${do_delta} \
            data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 16 --do_delta ${do_delta} \
            data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir} 
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 16 --do_delta ${do_delta} \
                data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
                ${feat_recog_dir}
    done
    echo "Feature Generation Successful"
fi

dict=data/${lang}_lang_char/${train_set}_word_units.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/${lang}_lang_char/
    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 -t word data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    echo "make json files"
    data2json.sh --feat ${feat_tr_dir}/feats.scp --trans_type word \
                 data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --trans_type word \
                 data/${train_dev} ${dict} > ${feat_dt_dir}/data.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp --trans_type word \
                     data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done
    echo "Dictionary and Json Data Preparation Successful"
fi

# you can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
fi
lmexpname=train_${lang}_rnnlm_${backend}_${lmtag}
lmexpdir=exp/${lmexpname}
mkdir -p ${lmexpdir}

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: LM Preparation"
    lmdatadir=data/local/wordlm_train
    mkdir -p ${lmdatadir}
    text2token.py -s 1 -n 1 -t word data/${train_set}/text | cut -f 2- -d" " > ${lmdatadir}/train.txt
    text2token.py -s 1 -n 1 -t word data/${train_dev}/text | cut -f 2- -d" " > ${lmdatadir}/valid.txt

    ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
        lm_train.py \
        --config ${lm_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --tensorboard-dir tensorboard/${lmexpname} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --resume ${lm_resume} \
        --dict ${dict}
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})_$(basename ${preprocess_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    nj=12
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]] || \
           [[ $(get_yaml.py ${train_config} model-module) = *conformer* ]] || \
           [[ $(get_yaml.py ${train_config} etype) = custom ]] || \
           [[ $(get_yaml.py ${train_config} dtype) = custom ]]; then
        if ${use_valbest_average}; then
            recog_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        if [ -z ${use_custom_model} ]; then
            average_checkpoints.py \
                        ${opt} \
                        --backend ${backend} \
                        --snapshots ${expdir}/results/snapshot.ep.* \
                        --out ${expdir}/results/${recog_model} \
                        --num ${n_average}
        else
            recog_model=${use_custom_model}
        fi
    fi
    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        if ${use_valbest_average}; then
            decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}_val${n_average}
        else
            decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}_last${n_average}
        fi
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --rnnlm ${lmexpdir}/rnnlm.model.best

        # score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}
        score_sclite.sh --wer true ${expdir}/${decode_dir} ${dict}
        python local/evaluate_decoded_captions.py ${expdir}/${decode_dir}/data.json \
            data/${rtask}/groundtruth_captions.txt > "${expdir}/${decode_dir}/caption_evaluation_summary.txt"
        echo "Evaluation metrics summary is saved to: ${expdir}/${decode_dir}/caption_evaluation_summary.txt"
        echo "Evaluation metrics individual results are saved to: ${expdir}/${decode_dir}/caption_evaluation_results.txt"

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
