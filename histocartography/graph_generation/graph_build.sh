# activate env 
conda activate histocartography
​
# Add root dir to python path
export PYTHONPATH="$PWD/../../:{$PYTHONPATH}"
​
# Create dir for output logs
mkdir -p runs
​
# Set input parameters
queue="prod.long"
​
# # start cell loop
# config=../config/cell_graph_model_config/cell_graph_model_config_gin.json
# # FEATURES_USED=(features_cnn_resnet50_mask_False_  features_cnn_resnet50_mask_True_  features_cnn_resnet34_mask_False_ features_cnn_resnet34_mask_True_)
# # FEATURES_USED=(features_hc_)
# FEATURES_USED=(features_cnn_resnet34_mask_False_)

# for f in "${FEATURES_USED[@]}"
# do
# 	echo "$f"
# 	echo "$c"
# 	bsub -R "rusage []" \
#              -J  "graph_generation" \
# 	     	 -o "runs/lsf_logs.%J.stdout" \
#              -e "runs/lsf_logs.%J.stderr" \
#              -q "$queue" \
#              "python build_graph.py --data_path /dataT/pus/histocartography/Data/BRACS_L --save_path /dataT/pus/histocartography/Data/BRACS_L --features_used $f --configuration $config"
#              sleep 0.1
# done

# start tissue loop 
config=../config/superpx_graph_model_config/superpx_graph_model_config_gin.json
# FEATURES_USED=(merging_hc_features_cnn_resnet50_mask_False_  merging_hc_features_cnn_resnet34_mask_False_  merging_hc_features_cnn_resnet50_mask_True_ merging_hc_features_cnn_resnet34_mask_True_)
# FEATURES_USED=(merging_hc_features_hc_)
FEATURES_USED=(merging_hc_features_cnn_resnet34_mask_False_)
for f in "${FEATURES_USED[@]}"
do
	echo "$f"
	echo "$c"
	bsub -R "rusage []" \
             -J  "graph_generation" \
	     	 -o "runs/lsf_logs.%J.stdout" \
             -e "runs/lsf_logs.%J.stderr" \
             -q "$queue" \
             "python build_graph.py --data_path /dataT/pus/histocartography/Data/BRACS_L/ --save_path /dataT/pus/histocartography/Data/BRACS_L --features_used $f --configuration $config"
             sleep 0.1
done


# # start assignment matrix 
# config=../config/multi_level_graph_model_config/multi_level_graph_model_config_0.json
# bsub -R "rusage []" \
#          -J  "graph_generation" \
#      	 -o "runs/lsf_logs.%J.stdout" \
#          -e "runs/lsf_logs.%J.stderr" \
#          -q "$queue" \
#          "python build_graph.py --data_path /dataT/pus/histocartography/Data/BRACS_L --save_path /dataT/pus/histocartography/Data/BRACS_L --features_used blabla --configuration $config"
