cd finetune
export COPPELIASIM_ROOT=$(pwd)/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export DISPLAY=:1.0
cd RLBench


pip uninstall -y opencv-python opencv-contrib-python
pip install  opencv-python-headless  
pip uninstall  -y opencv-python-headless      
pip install  opencv-python-headless   # in my machine , i have to repeat the installation process to avoid the error: "Could not find the Qt platform plugin 'xcb'"   
xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24 -ac'  python3 eval.py --model-folder PATH_TO_MODEL_FOLDER --eval-datafolder   PATH_TO_EVAL_DATAFOLDER \
 --tasks "all"  --eval-episodes 25 --log-name "debug_opensource_our_train" --device 0 --headless --model-name "model_99.pth" 
