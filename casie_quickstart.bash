#!/bin/bash

set -e

pushd dataset_processing

RAW_DIR="data/casie/data"

if [ ! -d "${RAW_DIR}" ]; then
    pushd data/casie
    set -x
    bash scripts/download_data.bash
    bash scripts/download_corenlp.bash
    bash run.bash
    set +x
    popd
fi

CONVERTED_DIR="converted_data/text2spotasoc/event/casie"
if [ ! -d "${CONVERTED_DIR}" ]; then
    set -x
    python -m uie_convert -format spotasoc -config data_config/event -output event -dataset casie
    python -m scripts.data_statistics -data converted_data/text2spotasoc/
    set +x
fi

popd

set -x

ln --force --symbolic dataset_processing/converted_data/ data

bash run_uie_finetune.bash -v -d 0 \
  -b 16 \
  -k 3 \
  --lr 1e-4 \
  --warmup_ratio 0.06 \
  -i event/casie \
  --epoch 50 \
  --spot_noise 0.1 \
  --asoc_noise 0.1 \
  -f spotasoc \
  --epoch 50 \
  --map_config config/offset_map/closest_offset_en.yaml \
  -m hf_models/uie-base-en \
  --random_prompt

set +x
