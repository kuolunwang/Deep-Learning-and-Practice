#!/usr/bin/env bash

echo "Install required libraries"
pip3 install virtualenv

virtualenv_name="DLP_Lab5"
VIRTUALENV_FOLDER=$(pwd)/${virtualenv_name}
virtualenv ${virtualenv_name}

source ${VIRTUALENV_FOLDER}/bin/activate
python3 -m pip install -r requirements.txt
deactivate
echo "alias DLP_Lab5='source ${VIRTUALENV_FOLDER}/bin/activate '" >> ~/.bashrc
source ~/.bashrc
