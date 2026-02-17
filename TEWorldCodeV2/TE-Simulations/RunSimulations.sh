# Iterates through each TE configuration folder, running all of the experiments and stores them in trace files.
# I implemented this as a bash script as it is simpler than Python for simpler commands

# Obtain all of the trial folders with glob within TE-Experiments
files=../../TE-Experiments/**

# For each of file, run the simulation function
for file in $files
do
    echo Running 3 simulations for $file...
    
    # Run the simulation within a subshell
    (
        cd $file
        python2.7 ../../TEWorldCodeV2/TESim.py 3
    )

    
    echo Completed simulation for $file.
done