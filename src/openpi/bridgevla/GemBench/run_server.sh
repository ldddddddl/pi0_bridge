cd  /PATH_TO_BRIDGEVLA/finetune/
export COPPELIASIM_ROOT=$(pwd)/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export DISPLAY=:1.0
cd ./GemBench
xvfb-run -a python3 server.py --port 13003 --model_epoch $1  --base_path $2
