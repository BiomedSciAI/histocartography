module load Anaconda3
conda activate cell_graph

# Add root dir to python path

export PYTHONPATH="$PWD/../:{$PYTHONPATH}"

# setup MLFLOW experiment 
source ~/.setup_MLflow.sh

# Create dir for output logs
mkdir -p ../runs

# Set input parameters
learning_rates=(0.001 0.01)
batch_size=(2 4)
queue="prod.med"
# Training loop
for bs in "${batch_size[@]}"
do
	for lr in "${learning_rates[@]}"
	do
		echo "$lr"
		echo "$bs"
		echo "$queue"
		bsub -R "rusage [ngpus_excl_p=1]" \
		    -J  "d-vae-training" \
		    -o "../runs/lsf_logs.%J.stdout" \
		    -e "../runs/lsf_logs.%J.stderr" \
		    -q "$queue" \
		    "/u/frd/.local/bin/mlflow run --no-conda histo-graph/ -P learning_rate=$lr -P batch_size=$bs -P epochs=100"
			sleep 1.0 
	done
done
