#!/bin/bash
function print_header(){
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
	echo $1
	printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
}

# Virtualenv
virtualenv venv
source venv/bin/activate

# Install dependencies
pip3 install -r requirements.txt

# Begin experiment
print_header "Training network"
python3.6 main.py \
--prefix "" \
--state_dim 2 \
--target 3 \
--render \
--mode "train"
