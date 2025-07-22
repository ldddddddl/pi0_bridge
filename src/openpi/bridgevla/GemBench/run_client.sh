cd  /PATH_TO_BRIDGEVLA/finetune/
sudo apt-get install -y jq
export COPPELIASIM_ROOT=$(pwd)/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export DISPLAY=:1.0
cd GemBench

seed=$1  # 200,300,400,500,600
epoch=$2  # model epoch number


for split in train test_l2 test_l3 test_l4; do
    json_file=./assets/taskvars_${split}.json

    taskvars=$(jq -r '.[]' "$json_file")
    for taskvar in $taskvars; do
        xvfb-run -a python3 client.py \
            --port 13003  \
            --output_file /PATH_TO_SAVE_RESULT_JSON/model_${epoch}/seed${seed}/${split}/result.json \
            --microstep_data_dir /PATH_TO_TEST_DATA/test_dataset/microsteps/seed${seed} \
            --taskvar "$taskvar"
    done
done

