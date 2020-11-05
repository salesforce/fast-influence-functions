# exit when any command fails
set -e

# Constants
REGISTRY="gcr.io/salesforce-research-internal"
REPO_BASE="hguo-scratchpad"

# Use Git commit SHA as the image tag
HeadCommitSHA=`git rev-parse --verify HEAD`
echo ${HeadCommitSHA}

# Using GCloud Builds
gcloud builds submit \
    --timeout 1000 \
    --tag ${REGISTRY}/${REPO_BASE}:${HeadCommitSHA} .

# Modify the tag in the related files
git checkout -- cluster/docker.run.sh  # this is important
sed -i -e 's@IMAGE_TAG@'"$HeadCommitSHA"'@' cluster/docker.run.sh

# Modify the tag in the related files
git checkout -- cluster/slurm.interactive.sh  # this is important
sed -i -e 's@IMAGE_TAG@'"$HeadCommitSHA"'@' cluster/slurm.interactive.sh

# Modify the tag in the related files
git checkout -- cluster/slurm.interactive.small.sh  # this is important
sed -i -e 's@IMAGE_TAG@'"$HeadCommitSHA"'@' cluster/slurm.interactive.small.sh

# Modify the tag in the related files
git checkout -- cluster/kube.jupyter.yaml  # this is important
sed -i -e 's@IMAGE_TAG@'"$HeadCommitSHA"'@' cluster/kube.jupyter.yaml

# Modify the tag in the related files
git checkout -- cluster/kube.experiments.yaml  # this is important
sed -i -e 's@IMAGE_TAG@'"$HeadCommitSHA"'@' cluster/kube.experiments.yaml

# Modify the tag in the related files
git checkout -- cluster/kube.jupyter.large.yaml  # this is important
sed -i -e 's@IMAGE_TAG@'"$HeadCommitSHA"'@' cluster/kube.jupyter.large.yaml

# Modify the tag in the related files
git checkout -- cluster/kube.jupyter.large-dshm.yaml  # this is important
sed -i -e 's@IMAGE_TAG@'"$HeadCommitSHA"'@' cluster/kube.jupyter.large-dshm.yaml

# Modify the tag in the related files
git checkout -- cluster/kube.jupyter.large.external.yaml  # this is important
sed -i -e 's@IMAGE_TAG@'"$HeadCommitSHA"'@' cluster/kube.jupyter.large.external.yaml
