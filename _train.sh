#!/bin/bash
function print_header(){
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
	echo $1
	printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
}

# Virtualenv
virtualenv venv -p python3
source venv/bin/activate

# Install dependencies
python3 -m pip install -r requirements.txt

# Begin experiment
print_header "Training network"
python3.6 main.py \
--prefix "" \
--state_dim 2 \
--rows 5 \
--cols 5 \
--x_rooms 2 \
--y_rooms 2 \
--target 3 \
--gui_width 800 \
--gui_height 800 \
