conda activate histocartography

# Add root dir to python path
export PYTHONPATH="$PWD/../../:{$PYTHONPATH}"

# setup MLFLOW experiment 
source ../_set_mlflow.sh

# Create dir for output logs
mkdir -p ../../runs

# Set input parameters
LEARNING_RATES=(0.01)
NUM_CLASSES=(2)
queue="prod.med"

# Training loop
for lr in "${LEARNING_RATES[@]}"
do
	for num_classes in "${NUM_CLASSES[@]}"
	do
		echo "$lr"
		echo "$num_classes"

		bsub -R "rusage [ngpus_excl_p=1]" \
		    -J  "explainer" \
		    -o "../../runs/lsf_logs.%J.stdout" \
		    -e "../../runs/lsf_logs.%J.stderr" \
		    -q "$queue" \
		    "python run_explainer.py -d /dataT/pus/histocartography/Data/pascale/ -conf ../../histocartography/config/explainer_config.json --epochs 1000  -l $lr --out_path /dataT/gja/histocartography/data/explanations/${num_classes}_classes/fold_4 --num_classes $num_classes"
		sleep 0.1 
	done
done