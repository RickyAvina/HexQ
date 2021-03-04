#!/bin/bash
function print_header(){
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
	echo $1
	printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
}

# Virtualenv
virtualenv venv -p python3
source venv/bin/activate

# OS specific
case "$OSTYPE" in
    darwin*) defaults write org.python.python ApplePersistenceIgnoreState NO ;;
esac

# Install dependencies
python3 -m pip install -r requirements.txt

# Begin experiment
print_header "Training network [GridEnv]"
python3 main.py \
--env "GridEnv-v0" \
--prefix "" \
--state_dim 2 \
--rows 5 \
--cols 5 \
--x_rooms 2 \
--y_rooms 2 \
--start 0 0 \
--target 3 \
--gui_width 800 \
--gui_height 800 \
--exploration_iterations 10000 \
--init_q -10.0 \
--lr 0.8 \
--gamma 1.0 \
--epsilon 0.9 \
--epsilon_decay 0.999 \
--min_epsilon 0.1 \
--max_steps 500 \
--binary_file "binaries/gridworld/mdps.pickle" \
--verbose \
--render 
