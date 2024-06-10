#!/bin/bash
set -e

# Image name and TAGE
export IMAGE_NAME="unscene3d"
export IMAGE_TAG="develop"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
docker build \
  --file ${DIR}/Dockerfile \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  --tag $IMAGE_NAME:$IMAGE_TAG \
  ${DIR}/..