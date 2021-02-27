#!/bin/bash
#
# Based mostly on the TED-LIUM and Switchboard recipe
#
# Copyright  2017  Johns Hopkins University (Author: Shinji Watanabe and Yenda Trmal)
# Apache 2.0
#

# Begin configuration section.
nj=10
decode_nj=10
stage=0
nnet_stage=-10
decode_stage=3
decode_only=false
num_data_reps=4
foreground_snrs="20:10:15:5:0"
background_snrs="20:10:15:5:0"
#enhancement=beamformit # gss or beamformit
enhancement=gss # gss or beamformit

# chime5 main directory path
# please change the path accordingly
chime5_corpus=

# End configuration section
. ./utils/parse_options.sh

. ./cmd.sh
. ./path.sh

#train_cmd=slurm.pl

if [ $decode_only == "true" ]; then
  stage=16
fi

set -e # exit on error

# chime6 data directories, which are generated from ${chime5_corpus},
# to synchronize audio files across arrays and modify the annotation (JSON) file accordingly
chime6_corpus=${PWD}/CHiME6
json_dir=${chime6_corpus}/transcriptions
audio_dir=${chime6_corpus}/audio
enhanced_dir=enhanced

if [[ ${enhancement} == *gss* ]]; then
  enhanced_dir=${enhanced_dir}_multiarray
  enhancement=${enhancement}_multiarray
fi

if [[ ${enhancement} == *beamformit* ]]; then
  enhanced_dir=${enhanced_dir}
  enhancement=${enhancement}
fi

test_sets="dev_${enhancement} eval_${enhancement}"
train_set=train_worn_simu_u400k

# This script also needs the phonetisaurus g2p, srilm, beamformit
./local/check_tools.sh || exit 1

###########################################################################
# We first generate the synchronized audio files across arrays and
# corresponding JSON files. Note that this requires sox v14.4.2,
# which is installed via miniconda in ./local/check_tools.sh
###########################################################################


if [ $stage -le 0 ]; then
  local/generate_chime6_data.sh \
    --cmd "$train_cmd" \
    ${chime5_corpus} \
    ${chime6_corpus}
fi


###########################################################################
# We prepare dict and lang in stages 1 to 3.
###########################################################################

if [ $stage -le 1 ]; then
  echo "$0:  prepare segment data..."
  # skip u03 and u04 as they are missing
  for mictype in worn u01 u02 u05 u06; do
    local/prepare_data_4ch.sh --mictype ${mictype} \
			  ${audio_dir}/train ${json_dir}/train data/train_${mictype}
  done
  for dataset in dev; do
    for mictype in worn ref; do
      local/prepare_data_4ch.sh --mictype ${mictype} \
			    ${audio_dir}/${dataset} ${json_dir}/${dataset} \
			    data/${dataset}_${mictype}
    done
  done
fi

exit 0;
