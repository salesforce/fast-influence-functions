# 0.1
Interpreting Large Pre-trained Language Models From The Training Set

### Major Changes
* Runnable docker image and Kubernetes config
* Added GPT2 model
* Added `notebooks/`
* Added wikitext-2 article sanity check experiment
* Added `utils/exact.py` for computing exact influences.

### Minor Changes
* Dockerfile
    - Added `wget`, `unzip`, `psmisc`, `vim`
    - Added `vimrc`

* Requirements
    - Added `transformers` (`v2.11.0`)


# 0.2
Switching gear towards downstream tasks

### Major Changes
* Have an initial running version
* Added `influence_utils.faiss_utils`
* Adding Faiss [dependency](https://github.com/kyamagu/faiss-wheels)
* Added support for `textattacks`
* Fixed a bug where `bert.pooler` parameters were filted during the gradient computation. The bug was caused by that `bert.pooler` parameters are not used in pre-trained LMs, and thus they were filted in the early stages of the codebase where the experiments are still performed on LMs. In the fine-tuned BERT models, however, `bert.pooler` parameters are used.
* Added nvidia-drivers to Dockerfile, and bumped `torch` base docker image from `1.5.0` to `1.5.1`


# 0.3

### Major Changes
* Splitting the implementation of sequential vs parallel influence calculation.
* Added customized Datasets.


# 0.4
### Major Changes
* Two `Dockerfile` and `cluster/docker.update.sh` for building in two environments
* Added `cluster/kube.experiments.yaml` for non-interactive experiments running.
* `experiments/mnli.py`: support running experiments on examples that with correct and incorrect predictions.
* `experiments/`: Added `remote_utils.py` for synchronizing files with a remote server.
* Removed deprecated `language_modeling.py`
* Added `run_experiments.py` and `scripts/run_experiments.sh` for non-interactive experiments running.
* Copied the GLUE data download script and paste them into `scripts/down_glue_data.py`.
* Edited the configuration files for running in different environments.
* Added demo. To run the demo, do `~/.local/bin/streamlit run run_demo.py`
* Added `graph_tool` to `Dockerfile.external`
* Support non-parallel settings in experiments.
* Added retraining experiments.
* Added public README.
* Added a new version of `experiments/hans.py`.
* Added `experiments/influence_helpers.py` for simplifying some codes.
* Make sure the return type is `bool` in `experiments/misc_utils.py:is_prediction_correct`.
* Added functions for returing helpful/harmful indices in `experiments/misc_utils.py` that also checks the number of negative/positive indices.
* Major cleanup and resign of `experiments/mnli.py`.
* Added a few notebooks.


### Minor Changes
* Added the missing `.gitignore`.
* Merged `cluster/docker.run.{gpu|cpu}.sh` into a single file used in custom environment
* Updated `cluster/docker.update.sh` to include recent file changes.
* `experiments/constants.py`: added more constants used in `experiments.remote_utils.py`
* `experiments/misc_utils.py`: added a few more useful tools.
* Updated `requirements.txt` with libraries used in `experiments/remote_utils.py`
* Added a few more configurations to `cluster/`
* Added a few more constants to `experiments/constants.py`
* Improved flexibility of the demo and visualization.
* Fix the bug on `np.bool_(True) is True => False`
* Minor changes to cluster scripts.
* Added function for helping results analysis in `experiments/visualization.py`.
* Added the option to specify x-range in `experiments/visualization_utils.py:plot_influences_distribution`.
* Removed the deprecated `influence_utils/nn_influence_utils.py:experimental_clip_gradient_norm_`.

# 0.5
### Major Changes
* Updated `experiments/hans.py` to new experiment-designs.
