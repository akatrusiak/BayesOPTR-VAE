#!/bin/bash

# inputs
facility="ariel"
path="ios-mws-hebt2-dragon"
start="HEBT:XCB0"
end="HEBT2:FC4"
measurement_device="HEBT2:FC4"
elements=('steerer' 'quad')
fileName="test"
jaya_server="beta" #beta #devel



# cloning example folder
orig="./build_utils/example/"
cp -r $orig "./build_utils/"$fileName"/" 

temp="./build_utils/"$fileName"/"
chmod -R +x $temp

python $temp"extract_beam_config.py" --name $fileName --starting_element $start --measurement_device $measurement_device --facility $facility --path $path --device_types "${elements[@]}" --jaya_server $jaya_server

#################################
######### Simulation ############
#################################

#simulation beam parameters
energy=500 # MeV
mass = 500 # MeV/c^2
charge=1 # e


# Added simulation flag
# Initialize flag
use_sim=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --sim)
      use_sim=true
      shift
      ;;
    *)
      shift
      ;;
  esac
done

# Check if --sim flag is present
if [ "$use_sim" = true ]; then
  echo "Preparing Simulation Environment"
  
  # Create simulation folder structure
  sim_dir="./simulation/transoptr/$fileName"
  mkdir -p $sim_dir
  
  # Copy XML files from temp folder to simulation folder
  cp $temp*.xml $sim_dir/
  echo "XML files copied to $sim_dir"

fi

# Remove temp folder
rm -r $temp