#!/bin/bash
set -e

mkdir "datasets"
cd "datasets"

# WebNLG
echo "======================================================"
echo "Downloading the WebNLG data..."
echo "======================================================"
git clone "https://github.com/ThiagoCF05/webnlg.git"
cd "webnlg"
git checkout "12ca34880b225ebd1eb9db07c64e8dd76f7e5784" 2>/dev/null
cd ..

# Cleaned E2E
echo "======================================================"
echo "Downloading the E2E data..."
echo "======================================================"
git clone "https://github.com/tuetschek/e2e-cleaning.git"
cd "e2e-cleaning"
git checkout "3cf74701a07a620b36bb63a6b771f02d9c1315a3" 2>/dev/null
cd ..

# DiscoFuse
echo "======================================================"
echo "Downloading the DiscoFuse data."
echo "======================================================"
echo "**Size of the archive to be downloaded is 1.6 GB.**"
read -p "Press ENTER to confirm..."
git clone "https://github.com/google-research-datasets/discofuse.git"
cd "discofuse"
wget "https://storage.googleapis.com/discofuse.appspot.com/discofuse_v1_wikipedia.zip"
unzip "discofuse_v1_wikipedia.zip"
rm "discofuse_v1_wikipedia.zip"
cd ..

# LaserTagger
echo "======================================================"
echo "Cloning the forked LaserTagger repository..."
echo "======================================================"
cd ..
git clone "https://github.com/kasnerz/lasertagger"
mv "lasertagger" "lasertagger_tf"

# BERT
echo "======================================================"
echo "Downloading the BERT model."
echo "======================================================"
echo "**Size of the archive to be downloaded is 385 MB.**"
read -p "Press ENTER to confirm..."
cd "lasertagger_tf"
mkdir "bert"
cd bert
wget "https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip"
unzip "cased_L-12_H-768_A-12.zip"
rm "cased_L-12_H-768_A-12.zip"
cd ../..

# TODO GPT-2
# TODO edited templates