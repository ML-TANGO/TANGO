#!/bin/sh

CURRENT=`pwd`
CURDATE=`date +"%Y%m%d%H%M%S"`
# src dir
BASE=${CURRENT%/*}
SSRC=${BASE}/Server
CSRC=${BASE}/Client
MSRC=${BASE}/Model
echo "BASEDIR=${BASE}"

# release dir
RELEASE=${BASE}/release
SRELEASE=${RELEASE}/server
CRELEASE=${RELEASE}/client
MRELEASE=${RELEASE}/model
echo "RELEASEDIR=${RELEASE}"

# clear release dir
echo "Clear release directory..."
rm -rf ${RELEASE}

# create release dir
echo "Create release directory..."
mkdir -p ${SRELEASE}
mkdir -p ${CRELEASE}
mkdir -p ${MRELEASE}

# build server
echo "Build server..."
cd ${SSRC}
npm install
npm run build
cp -R ${SSRC}/dist ${SRELEASE}
cp -R ${SSRC}/node_modules ${SRELEASE}
find ${SRELEASE}/dist/config ! -name "server-release.json" -type f -delete
mv ${SRELEASE}/dist/config/server-release.json ${SRELEASE}/dist/config/server.json

# build client
echo "Build client..."
cd ${CSRC}
npm install
npm run build:prod
cp -R ${CSRC}/dist/* ${CRELEASE}

# copy model
echo "Build model..."
cd ${MSRC}
cp -R ${MSRC}/* ${MRELEASE}

# packing
#echo "Packing..."
#cd ${BASE}
#tar cvf release_${CURDATE}.tar ${RELEASE}
#gzip release_${CURDATE}.tar
echo "Done"