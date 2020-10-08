# conda activate histocartography
module load Miniconda3

# Add root dir to python path
export PYTHONPATH="$PWD/../../:{$PYTHONPATH}"

# setup MLFLOW experiment 
source ../_set_mlflow.sh

# Create dir for output logs
mkdir -p ../../runs

# Set input parameters
BASE_CONFIG="explain_config/3_class_scenario/"
# ALL_CONFIG_FILES=($(ls ../../histocartography/config/${BASE_CONFIG} | grep .json))
# ALL_CONFIG_FILES=("cnn_model_config_gradpp.json" "cnn_model_config_deeplift.json" "cnn_model_config_grad.json")
ALL_CONFIG_FILES=("cell_graph_model_config_lrp.json" "cell_graph_model_config_grad.json")
# ALL_CONFIG_FILES=("cell_graph_model_config_gnn_explainer.json")
SPLITS=("test")
queue="prod.med"

for split in "${SPLITS[@]}"
	do
	for conf in "${ALL_CONFIG_FILES[@]}"
	do
		bsub -R "rusage [ngpus_excl_p=1]" \
		    -J  "explainer" \
		    -o "../../runs/lsf_logs.%J.stdout" \
		    -e "../../runs/lsf_logs.%J.stderr" \
		    -q "$queue" \
		    "python run_explainer.py -d /dataT/pus/histocartography/Data/BRACS_L/ -conf ../../histocartography/config/$BASE_CONFIG/$conf --split $split"
		sleep 0.1 
	done
done