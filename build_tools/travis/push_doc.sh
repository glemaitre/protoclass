#!/bin/bash

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
git clone "git@github.com:$USERNAME/"$PROJECT".git";

cd $PROJECT
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

# Get the deploy key by using Travis's stored variables to decrypt deploy_key.enc
ENCRYPTED_KEY_VAR="encrypted_${ENCRYPTION_LABEL}_key"
ENCRYPTED_IV_VAR="encrypted_${ENCRYPTION_LABEL}_iv"
ENCRYPTED_KEY=${!ENCRYPTED_KEY_VAR}
ENCRYPTED_IV=${!ENCRYPTED_IV_VAR}
openssl aes-256-cbc -K $ENCRYPTED_KEY -iv $ENCRYPTED_IV -in deploy_key.enc -out deploy_key -d
chmod 600 deploy_key
eval `ssh-agent -s`
ssh-add deploy_key

git push -f origin gh-pages
if [ $? -ne 0 ]; then
    echo "Pushing docs failed"
    echo
    exit 1
fi
