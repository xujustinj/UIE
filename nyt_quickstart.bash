#!/bin/bash

set -e
set -x

pushd dataset_processing

RAW_DIR="data/NYT-multi"
URL="https://raw.githubusercontent.com/yubowen-ph/JointER/master/dataset/NYT-multi/data"
if [ ! -d "${RAW_DIR}" ]; then
    mkdir --parent "${RAW_DIR}"
    wget -P "${RAW_DIR}" "${URL}/train.json"
    wget -P "${RAW_DIR}" "${URL}/dev.json"
    wget -P "${RAW_DIR}" "${URL}/test.json"
fi

CONVERTED_DIR="converted_data/text2spotasoc/relation/NYT"
if [ ! -d "${CONVERTED_DIR}" ]; then
    python -m uie_convert -format spotasoc -config data_config/relation -output relation -dataset NYT-multi
    python -m scripts.data_statistics -data converted_data/text2spotasoc/
fi

# for seed in 1 2 3 4 5 6 7 8 9 10; do
    # ONLY NEEDED FOR LOW-RESOURCE EXPERIMENTS, WASTE OF DISK SPACE OTHERWISE
    # RATIO_TARGET="converted_data/text2spotasoc/relation/NYT_ratio/seed${seed}"
    # if [ ! -d "${RATIO_TARGET}" ]; then
    #     python -m scripts.sample_data_ratio -seed ${seed} \
    #         -src "${CONVERTED_DIR}" \
    #         -tgt "${RATIO_TARGET}"
    # fi

    # ONLY NEEDED FOR FEW-SHOT EXPERIMENTS, WASTE OF DISK SPACE OTHERWISE
    # SHOT_TARGET="converted_data/text2spotasoc/relation/NYT_shot/seed${seed}"
    # if [ ! -d "${SHOT_TARGET}" ]; then
    #     python -m scripts.sample_data_shot -seed ${seed} \
    #         -src "${CONVERTED_DIR}" \
    #         -tgt "${SHOT_TARGET}" \
    #         -task relation
    # fi
# done

popd

ln --force --symbolic dataset_processing/converted_data/ data

bash run_uie_finetune.bash -v -d 0 \
  -b 16 \
  -k 3 \
  --lr 1e-4 \
  --warmup_ratio 0.06 \
  -i relation/NYT \
  --epoch 50 \
  --spot_noise 0.1 \
  --asoc_noise 0.1 \
  -f spotasoc \
  --epoch 50 \
  --map_config config/offset_map/closest_offset_en.yaml \
  -m hf_models/uie-base-en \
  --random_prompt
