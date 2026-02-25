# Iterates through each TE configuration folder, running all of the experiments and stores them in trace files.
# I implemented this as a bash script as it is simpler than Python for simpler commands

# Obtain all of the trial folders with glob within TE-Experiments
files=../../TE-Experiments/**
early_exit=0


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

        # Enables early exit
        if [ $? -eq 1 ]
        then 
            early_exit=1
            break
        fi
        echo Completed simulation for $file.
    done

    if [ $early_exit -eq 1 ]
    then
        break
    fi
    echo Pass-through $i finished.
done

# Don't print out statement if early exit was triggered
if [ $early_exit -ne 1 ]
then
    echo Completed all simulations!
fi