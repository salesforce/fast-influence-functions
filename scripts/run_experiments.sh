# Setup Remote server directory to save outputs
export REMOTE_BASE_DIR="/data/home/guest/Experiments/20200914"

# Setup Weight And Bias
export WANDB_API_KEY="25a0cba5a031d8071bb6fcd352cdb659ba152a92"

# Run Experiments!
python ./run_experiments.py $1
