#!/bin/bash

# Function to check if Anaconda is installed
check_anaconda() {
    if ! command -v conda &> /dev/null; then
        echo "Anaconda is not installed. Please install Anaconda first."
        exit 1
    fi
}

# Function to check if the CH environment exists
check_ch_environment() {
    conda env list | grep -q "^CH "
}

# Function to activate the CH environment
activate_ch() {
    echo "Activating CH environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate CH
    if [ $? -eq 0 ]; then
        echo "CH environment activated successfully."
    else
        echo "Failed to activate CH environment."
        exit 1
    fi
}

# Function to create the CH environment from the YAML file
create_ch_environment() {
    echo "Creating CH environment from env/CH.yml..."
    conda env create -f env/CH.yml
    if [ $? -eq 0 ]; then
        echo "CH environment created successfully."
        activate_ch
    else
        echo "Failed to create CH environment."
        exit 1
    fi
}

# Function to run a preprocessing script
run_preprocessing_script() {
    script_name=$1
    echo "Running $script_name..."
    python "$script_name"
    if [ $? -eq 0 ]; then
        echo "$script_name completed successfully."
    else
        echo "$script_name failed."
        exit 1
    fi
}

# Main script
check_anaconda

if check_ch_environment; then
    activate_ch
else
    if [ -f env/CH.yml ]; then
        create_ch_environment
    else
        echo "env/CH.yml file not found. Cannot create CH environment."
        exit 1
    fi
fi

# Change directory to the scripts folder
echo "Changing directory to scripts folder..."
cd scripts || { echo "Failed to change directory to scripts. Exiting."; exit 1; }

# Run the preprocessing scripts
run_preprocessing_script "1_preprocessing_LBC_cohort_1.py"
run_preprocessing_script "2_preprocessing_LBC_cohort_2.py"
run_preprocessing_script "3_preprocessing_sardiNIA.py"
run_preprocessing_script "4_preprocessing_WHI.py"
run_preprocessing_script "5_LIFT_LBC_cohort_1.py"
run_preprocessing_script "6_LIFT_LBC_cohort_2.py"
run_preprocessing_script "7_clonal_fit.py"

echo "All preprocessing scripts completed successfully."

# Change back to the original directory
cd ..