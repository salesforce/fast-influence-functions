# exit when any command fails
set -e

# Constants
REPO_BASE="fast-influence-functions"
DIRECTORY_BASE="fast-influence-functions"


# Use Git commit SHA as the image tag
HeadCommitSHA=`git rev-parse --verify HEAD`
echo ${HeadCommitSHA}

# Remove, and re-build container
# docker image rm hanguo97/${REPO_BASE}
cd /proj/bansallab/users/han/DLResearch

# Using BuiltKit
# More: https://docs.docker.com/develop/develop-images/build_enhancements/
DOCKER_BUILDKIT=1 docker build \
    -t ${REPO_BASE} \
    -f ${DIRECTORY_BASE}/Dockerfile \
    --secret id=wandb_apikey,src=secrets/wandb.apikey.secret  .

# Push to the registry
docker tag ${REPO_BASE} hanguo97/${REPO_BASE}:${HeadCommitSHA}
docker push hanguo97/${REPO_BASE}:${HeadCommitSHA}

# Modify the tag in the related files
git checkout -- cluster/docker.run.gpu.sh  # this is important
sed -i -e 's@IMAGE_TAG@'"$HeadCommitSHA"'@' cluster/docker.run.gpu.sh

# Modify the tag in the related files
git checkout -- cluster/docker.run.cpu.sh  # this is important
sed -i -e 's@IMAGE_TAG@'"$HeadCommitSHA"'@' cluster/docker.run.cpu.sh

# Modify the tag in the related files
git checkout -- cluster/kube.jupyter.yaml  # this is important
sed -i -e 's@IMAGE_TAG@'"$HeadCommitSHA"'@' cluster/kube.jupyter.yaml

# Modify the tag in the related files
git checkout -- cluster/kube.jupyter.large.yaml  # this is important
sed -i -e 's@IMAGE_TAG@'"$HeadCommitSHA"'@' cluster/kube.jupyter.large.yaml

# Modify the tag in the related files
git checkout -- cluster/kube.jupyter.large-dshm.yaml  # this is important
sed -i -e 's@IMAGE_TAG@'"$HeadCommitSHA"'@' cluster/kube.jupyter.large-dshm.yaml
