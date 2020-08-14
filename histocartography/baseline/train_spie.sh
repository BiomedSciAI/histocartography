
module purge
source ../../config.sh

# Create dir for output logs
mkdir -p runs

SPLITS=(1)
# PATCH_SIZE=("10x" "20x" "40x")
PATCH_SIZE=("10x")
# CLASS_SPLIT=("benignVSpathologicalbenignVSudhVSadhVSfeaVSdcisVSmalignant" "benignVSpathologicalbenign+udhVSadh+feaVSdcis+malignant" "benign+pathologicalbenign+udh+adh+fea+dcisVSmalignant" "benign+pathologicalbenign+udhVSadh+fea+dcis" "benignVSpathologicalbenign+udh" "pathologicalbenignVSudh" "adh+feaVSdcis" "adhVSfea")
# CLASS_SPLIT=("benignVSpathologicalbenign+udhVSadh+feaVSdcis+malignant" "benign+pathologicalbenign+udh+adh+fea+dcisVSmalignant" "benign+pathologicalbenign+udhVSadh+fea+dcis" "benignVSpathologicalbenign+udh")
CLASS_SPLIT=("benignVSpathologicalbenignVSudhVSadhVSfeaVSdcisVSmalignant")
# REPEAT=(0 1 2)
REPEAT=(0)

# debug parameters 
# PATCH_SIZE=("10x")
# CLASS_SPLIT=("benignVSpathologicalbenignVSudhVSadhVSfeaVSdcisVSmalignant")
# REPEAT=(0)

queue="prod.short"

# setup MLFLOW experiment 
source ../../experiments/_set_mlflow.sh

# export experiment 
export MLFLOW_EXPERIMENT_NAME=gja_bracs_l_spie
# mlflow experiments create --artifact-location s3://mlflow -n ${MLFLOW_EXPERIMENT_NAME}

for repeat in "${REPEAT[@]}"
do
	for patch_size in "${PATCH_SIZE[@]}"
	do
		for class_split in "${CLASS_SPLIT[@]}"
		do
			for split in "${SPLITS[@]}"
			do
				echo "$lr"
				echo "$bs"
				echo "$conf"
				bsub -R "rusage [ngpus_excl_p=1]" \
				    -J  "SPIE_baseline" \
				    -o "runs/lsf_logs.%J.stdout" \
				    -e "runs/lsf_logs.%J.stderr" \
				    -q "$queue" \
				    "python main.py --split $split --is_extraction False --mode gnn_merge --num_epochs 100 --patch_size $patch_size --patch_scale $patch_size --class_split $class_split --model_type base_pt_th0_cv --data_param dataT --gpu 0 --in_ram True"
				sleep 0.1 
			done 
		done
	done
done
