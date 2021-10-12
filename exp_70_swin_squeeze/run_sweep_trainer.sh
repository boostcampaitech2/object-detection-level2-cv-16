#!/bin/sh

run_trainer_with_config()
{
    if [ -f ./exp_config/default_config.py ];
    then
        wandb sweep ./sweep_config/sweep_config.yaml

        read -p "Let me know the sweep id ::: " sweep_id
        nohup wandb agent ${sweep_id} & # > sweep_log.log &

        #tail -f sweep_log.log

    else
        echo "========================================"
        echo "=                                      ="
        echo "= !!! WARNING !!!                      ="
        echo "=                                      ="
        echo "= There is no available config file.   ="
        echo "= Please set a integrated config file  ="
        echo "= as './exp_config/default_config.py'. ="
        echo "=                                      ="
        echo "= Or use this code to merge your       ="
        echo "= costum config files.                 ="
        echo "= python ./exp_utils/merge_config.py   ="
        echo "========================================"
    fi

}

if [ ! -d ./wandb ]
then
    echo "_________________________________________"
    echo " Please check wandb state. you should run"
    echo " [ wandb init ] on your experiment directory."
fi
    
    
if [ -d ./work-dir ]
then
    echo "_________________________________________"
    echo "There is already existed work-dir."
    echo "This work may overwrite your experiments."
    echo "Are you sure to run the new sweep agent?"
    read -p "[y/n] : " answer
    if [ ${answer} = "y" ]
    then
        run_trainer_with_config
    else
        echo "End of shell script by 'n'"
    fi
else
    run_trainer_with_config
fi
