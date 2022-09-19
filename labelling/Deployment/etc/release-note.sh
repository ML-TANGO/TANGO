#!/bin/sh

GITLAB_PERSONAL_TOKEN=9yCNhG22yAgsx5X9RWfK
GITLAB_PROJECT_ID=$1
GITLAB_API_ENDPOINT=https://www.wedalab.com/api/v4
#TARGET_BRANCH=$2

docker run \
	--name release-note-${GITLAB_PROJECT_ID} \
	-e GITLAB_API_ENDPOINT=${GITLAB_API_ENDPOINT} \
	-e GITLAB_PERSONAL_TOKEN=${GITLAB_PERSONAL_TOKEN} \
	-e GITLAB_PROJECT_ID=${GITLAB_PROJECT_ID} \
	00freezy00/gitlab-release-note-generator:latest
	#-e TARGET_BRANCH=${TARGET_BRANCH} \
	#-e TARGET_TAG_REGEX=sampleReg \
	#00freezy00/gitlab-release-note-generator:latest