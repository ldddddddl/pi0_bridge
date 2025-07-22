'''
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
Adapted from https://github.com/vlc-robot/robot-3dlotus/blob/main/challenges/server.py

Author: Peiyan Li
Email: peiyan.li@cripac.ia.ac.cn
'''
import argparse
import msgpack_numpy
msgpack_numpy.patch()

import sys
from flask import Flask, request

from actioner import MyActioner

def main(args):
    app = Flask(__name__)
    actioner = MyActioner(args.base_path,args.model_epoch)

    @app.route('/predict', methods=['POST'])
    def predict():
        '''
        batch is a dict containing:
            instruction: str
            obs_state_dict: observations from genrobo3d.rlbench.environments.RLBenchEnv 
        '''
        # data = request.data
        data = request.get_data()
        batch = msgpack_numpy.unpackb(data, raw=False)

        action = actioner.predict(**batch)
        action = msgpack_numpy.packb(action)
        return action
    
    app.run(host=args.ip, port=args.port, debug=args.debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Actioner server')
    parser.add_argument('--ip', type=str, default="localhost")
    parser.add_argument('--port', type=int, default=13003)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--base_path', type=str, default="/PATH_TO_MODEL_FOLDER")
    parser.add_argument('--model_epoch', type=int, default=40) 
    args = parser.parse_args()
    main(args)
