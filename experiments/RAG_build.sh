module load Anaconda3
conda activate cell_graph

# Add root dir to python path

export PYTHONPATH="$PWD/../:{$PYTHONPATH}"

# Create dir for output logs
mkdir -p ../runs

queue="prod.long"
# Training loop
echo "$queue"
bsub -R "rusage [ngpus_excl_p=1]" \
-J  "RAG building" \
-o "../runs/lsf_logs.%J.stdout" \
-e "../runs/lsf_logs.%J.stderr" \
-q "$queue" \
"python3 histo-graph/RAG_graph_building.py -s /dataT/frd/data_test_dummy/dgl_graphs/"
