#!/bin/bash

set -o errexit

cd $TRAVIS_BUILD_DIR

# We need to move to the repo where the doc was generated
cd $TRAVIS_BUILD_DIR/../$DOC_REPO/html

# Create a nojekyll file
touch .nojekyll

# Create a tempory folder in order to copy all the doc in the already
# tracked repo
cd $TRAVIS_BUILD_DIR/../
rm -rf tmp
mkdir tmp
cd tmp

# Clone the repository to push the doc later
git clone "https://github.com/glemaitre/protoclass.git"

cd $PROJECT
git remote set-url origin "https://glemaitre:${GH_TOKEN}@github.com/glemaitre/${PROJECT}.git"
git branch gh-pages
git checkout -f gh-pages
git reset --hard origin/gh-pages
git clean -dfx

# Copy the new build docs
cp -R $TRAVIS_BUILD_DIR/../$DOC_REPO/html/* ./

git config --global user.email $EMAIL
git config --global user.name $USERNAME
git add -f ./
git commit -m "Push the doc automatically"

echo "Done with the commit, let's push"

git push --force origin gh-pages
