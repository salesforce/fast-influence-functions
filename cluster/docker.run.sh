REPO_BASE="fast-influence-functions"
DOCKER_TAG="IMAGE_TAG"

nvidia-docker run -ti --rm \
    --network hanguo \
    --user 259446:3406 \
    -v /proj/bansallab/users/:/nlp \
    hanguo97/${REPO_BASE}:${DOCKER_TAG}
