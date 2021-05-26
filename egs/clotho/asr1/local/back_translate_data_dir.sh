#!/bin/bash

# Perform back-translation for data dir

. ./path.sh || exit 1

tgt_lang=en
inter_lang=es

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1

src_data_dir=$1

if [ $# -ne 1 ]; then
  echo "Usage: $0 [options] <data_dir>"
  echo "Options:"
  echo "  --tgt_lang: Target language (default=${tgt_lang})"
  echo "  --inter_lang: Intermediate languagte (default=${inter_lang})"
  exit 1
fi

set -euo pipefail

utils/copy_data_dir.sh "${src_data_dir}" "${src_data_dir}_${inter_lang}2${tgt_lang}"
local/back_translate_text.py \
    --tgt_lang "${tgt_lang}" \
    --inter_lang "${inter_lang}" \
    "${src_data_dir}/text" \
   "${src_data_dir}_${inter_lang}2${tgt_lang}/text"

echo "Sucessfully finished back-tranlation."
