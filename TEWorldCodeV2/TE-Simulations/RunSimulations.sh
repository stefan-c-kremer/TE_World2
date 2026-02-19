# Iterates through each TE configuration folder, running all of the experiments and stores them in trace files.
# I implemented this as a bash script as it is simpler than Python for simpler commands

# Obtain all of the trial folders with glob within TE-Experiments
files=../../TE-Experiments/**


echo Running 3 simulations for each parameters.py configuration, via three separate pass-throughs of all of the configurations.

for i in {1..3}
do
    echo Pass-through $i starting...
    # For each file, run the simulation function
    for file in $files
    do    
        # Run the simulation within a subshell
        (
            cd $file
            python2.7 ../../TEWorldCodeV2/TESim.py 1
        )

        
        echo Completed simulation for $file.
    done
    echo Pass-through $i finished.
done

echo Completed all simulations!