#!/bin/bash

module load Miniconda3
queue="prod.short"

# setup MLFLOW experiment
source ~/.setup_MLflow.sh

# export experiment
export MLFLOW_EXPERIMENT_NAME=pus_nuclei_classification
#mlflow experiments create --artifact-location s3://mlflow -n ${MLFLOW_EXPERIMENT_NAME}

# Set input parameters
ARCH=("resnet32" "resnet44" "resnet56" "resnet110")
EPOCHS=(100)
BATCH_SIZES=(128 256)
LEARNING_RATES=(0.1 0.01 0.001)
PRETRAINED=(True False)
FINETUNE=(True False)
WEIGHTED_LOSS=(True False)

# Create dir for output logs
mkdir -p ./runs


# Training loop
for arch in "${ARCH[@]}"
do
  for epoch in "${EPOCHS[@]}"
  do
	for bs in "${BATCH_SIZES[@]}"
	do
      for lr in "${LEARNING_RATES[@]}"
      do
        for pt in "${PRETRAINED[@]}"
        do
          for ft in "${FINETUNE[@]}"
          do
            for wl in "${WEIGHTED_LOSS[@]}"
            do
              echo "$arch"
              echo "$epoch"
              echo "$bs"
              echo "$lr"
              echo "$pt"
              echo "$ft"
              echo "$wl"
              bsub -R "rusage [ngpus_excl_p=1]" \
                   -n 1 \
                   -J  "nuclei_training" \
                   -o "./runs/lsf_logs.%J.stdout" \
                   -e "./runs/lsf_logs.%J.stderr" \
                   -q "$queue" \
                   "python -u main.py --data-param=dataT --mode=train --arch=$arch --epochs=$epoch --batch-size=$bs --lr=$lr --pretrained=$pt --finetune=$ft --weighted-loss=$wl"
              sleep 0.1
            done
          done
        done
      done
	done
  done
done





