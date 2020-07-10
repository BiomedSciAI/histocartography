conda activate histocartography

# Add root dir to python path
export PYTHONPATH="$PWD/../../:{$PYTHONPATH}"

# setup MLFLOW experiment 
source ../_set_mlflow.sh

# export experiment 
export MLFLOW_EXPERIMENT_NAME=gja_bracs_l_hact_5
# mlflow experiments create --artifact-location s3://mlflow -n ${MLFLOW_EXPERIMENT_NAME}

# Create dir for output logs
mkdir -p ../../runs

# Set input parameters
LEARNING_RATES=(0.005 0.001 0.0005)
BATCH_SIZES=(8 16 32)
BASE_CONFIG="multi_level_graph_model_config"
ALL_CONFIG_FILES=($(ls ../../histocartography/config/${BASE_CONFIG} | grep .json))
queue="prod.med"

# Training loop
for bs in "${BATCH_SIZES[@]}"
do
	for lr in "${LEARNING_RATES[@]}"
	do
		for conf in "${ALL_CONFIG_FILES[@]}"
		do
			echo "$lr"
			echo "$bs"
			echo "$conf"
			bsub -R "rusage [ngpus_excl_p=1]" \
			    -J  "HG_training" \
			    -o "../../runs/lsf_logs.%J.stdout" \
			    -e "../../runs/lsf_logs.%J.stderr" \
			    -q "$queue" \
			    "python train_histo_graph_cv.py --data_path /dataT/pus/histocartography/Data/BRACS_L/ -conf ../../histocartography/config/$BASE_CONFIG/$conf -l $lr -b $bs --epochs 100 --in_ram"
			sleep 0.1 
		done 
	done
done
