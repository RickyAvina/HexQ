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
print_header "Training network [Taxi]"
python3 main.py \
--env "Taxi-v4" \
--prefix "" \
--state_dim 4 \
--binary_file "binaries/taxi/mdps.pickle"
