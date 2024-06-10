set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source ${DIR}/build.sh

# Run the interactive script
docker run --interactive --tty --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --volume ${DIR}/..:/UnScene3D \
    --volume /media/data/Datasets/UnScene3D:/UnScene3D/data \
    --volume /media/data/Datasets/ScanNet:/UnScene3D/data/ScanNet \
    $IMAGE_NAME:${IMAGE_TAG} /bin/bash