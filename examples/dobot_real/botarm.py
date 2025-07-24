import random
import socket
import re
import time
import os
import threading 
from datetime import datetime
# from pynput import keyboard
from scipy.spatial.transform import Rotation as R
import numpy as np
from pymodbus.client import ModbusTcpClient
import transforms3d
import random
import struct

import glob
import logging
from logging.handlers import RotatingFileHandler

log_directory = "logs/"

if not os.path.exists(log_directory):
    try:
        os.makedirs(log_directory)
    except Exception as e:
        raise

logger = logging.getLogger("dobot_log")
logger.setLevel(logging.INFO)

log_filename = f'dobot_{time.strftime("%Y-%m-%d_%H-%M-%S")}.log'
# log_filename = f'dobot_{time.time()}.log'
file_handler = RotatingFileHandler(os.path.join(log_directory, log_filename),maxBytes= 1024*1024*200,backupCount=10)
# console_handler = logging.StreamHandler()  # 添加终端log输出

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
# console_handler.setFormatter(formatter)
# console_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
# logger.addHandler(console_handler)


class Server():
    def __init__(self, ip, port, host, app_port):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.app = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp = None
        self.host = host
        self.app_port = app_port
        self.baudrate = 115200
        self.modbus = None
        self.modbusRTU = None
        self.timestamp = None
        self.signal = {'replay':False,'claw_open':False,'claw_close':False, 'set_drag':False, 'reset_drag':False}
        # self.init_com()
    def init_com(self, bot):
        msg_modbus = f'Create("{self.ip}", 502, 2)'
        msg_modbusrtu = f'Create(1, {self.baudrate}, "N", 8, 1)'
        self.modbus = self._parse_id(bot.send_modbus_command('Modbus', msg_modbus))
        self.modbusRTU = self._parse_id(bot.send_modbus_command('ModbusRTU', msg_modbusrtu))

    def _parse_id(self, response):
        match = re.search(r'\{(.+?)\}', response)

        if match:
            numbers_str = match.group(1)
            return int(numbers_str)
    def start_server(self):
        """启动服务器"""
        self.app = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.app.bind((self.host, self.app_port))
        self.app.listen(5)  # 最大等待连接数为5
        print(f"Server listening on {self.host}:{self.app_port}")

        try:
            while True:
                self.tcp, addr = self.app.accept()
                print(f"Accepted connection from {addr[0]}:{addr[1]}")
                # 创建新线程处理客户端
                client_handler = threading.Thread(
                    target=self.handle_client,
                    args=(self.tcp, )
                )
                client_handler.daemon = True
                client_handler.start()
        except KeyboardInterrupt:
            print("Stopping server...")
            self.app.close()
            self.tcp.close()

    def handle_client(self):
        """处理客户端的通信"""
        # global replay_motion
        # global claw_open
        # global claw_close
        try:
            while True:
                data = self.sock.recv(1024).decode().strip()
                if not data:
                    # 客户端断开连接
                    print("Client disconnected")
                    break
                print(f"Received data: {data}")
                if data == 'replay':
                    self.signal['replay'] = True
                elif data == 'open':
                    self.signal['claw_open'] = True
                elif data == 'close':
                    self.signal['claw_close'] = True
                elif data == 'set':
                    self.signal['set_drag'] = True
                elif data == 'reset':
                    self.signal['reset_drag'] = True
                self.sock.send("Message received".encode())
        except socket.error as e:
            print(f"Socket error: {e}")
        finally:
            # 关闭客户端 socket
            self.sock.close()
class Point():
    def __init__(self,  position: list, quaternion: list, claw, gripper_thres=None, name=None):
        self.name = name  
        self.position = position
        self.quaternion = quaternion
        self.euler = None
        self.timestamp = None
        self.position_quaternion_claw = None
        self.claw = claw
        self.gripper_thres = gripper_thres
        self.__position_to_string()
        self.__quaternion_to_euler()
        self.__get_position_and_quaternion()
    def get_timestamp(self):
        dt = datetime.now()
        micro = dt.microsecond // 1000
        self.timestamp = dt.strftime(f"%Y-%m-%d_%H-%M-%S-{micro:03d}")
        return self.timestamp
    def __position_to_string(self):
        if self.position is None:
            return None
        # self.position_str = f"{{{self.position['x']:.4f},{self.position['y']:.4f},{self.position['z']:.4f},\
        #     {self.position['rx']},{self.position['ry']},{self.position['rz']}}}"
        # noise = np.random.normal(0, 0.01, size=3)  # 生成均值为0，标准差为0.001的3维随机噪声
        # self.position= self.position + noise
        self.position = f"{{{self.position[0]:.4f},{self.position[1]:.4f},{self.position[2]:.4f},"
        
        # return self.position_str
    # 四元数转欧拉角
    def __quaternion_to_euler(self):

        # 将列表转换为 numpy 数组（如果需要）
        self.quaternion = np.array(self.quaternion)

        # 确保四元数是单位四元数
        self.quaternion = self.quaternion / np.linalg.norm(self.quaternion)

        # 创建 Rotation 对象
        rotation = R.from_quat(self.quaternion)

        # 转换为欧拉角（弧度）
        euler_angles_rad = rotation.as_euler('xyz')


        # 转换为欧拉角（角度）
        euler_angles_deg = np.degrees(euler_angles_rad)

        # noise = np.random.normal(0, 0.01, size=3)  # 生成均值为0，标准差为0.001的3维随机噪声
        # euler_angles_deg = euler_angles_deg+ noise

        self.euler = f"{euler_angles_deg[0]:.4f},{euler_angles_deg[1]:.4f},{euler_angles_deg[2]:.4f}}}"

        return euler_angles_deg



  
    def __get_position_and_quaternion(self):
        self.position_quaternion_claw = self.position + self.euler

class DobotController():
    def __init__(self, sock):
        self.sock = sock
        self.modbus = None
        self.modbusRTU = None
        # self._initialize()
        self.current_pose = None
        

    def _initialize(self, server):
        self.send_command("PowerOn()")
        time.sleep(1)
        self.send_command("EnableRobot()")
        self.send_command("ClearError()")
        self.modbus = server.modbus
        self.modbusRTU = server.modbusRTU
    def point_control(self, point = None):
        print("+++++++",point.position_quaternion_claw)
        if point != None:
            pose = point.position_quaternion_claw
            pose_value = [float(v) for v in pose.strip('{}').split(',')]
            if self.current_pose is not None:
                cur_pose_value = [float(v) for v in self.current_pose.strip('{}').split(',')]
                z_cur = cur_pose_value[2]
                z_next = pose_value[2]

                h = 100.
                if z_cur < 75. and z_next >75. :
                    pose_value_1 = cur_pose_value
                    pose_value_1[2] = h
                    pose_1 = "{" + ",".join(map(str, pose_value_1)) + "}"
                    print("----- mid point : ", pose_1)
                    self.current_pose = pose_1
                    self.move_joint(pose_1)
                # if z_cur < 200. and z_next < 75:
                #     pose_value_1 = cur_pose_value
                #     pose_value_1[2] = 300.
                #     pose_1 = "{" + ",".join(map(str, pose_value_1)) + "}"
                #     print("----- mid point : ", pose_1)
                #     self.current_pose = pose_1
                #     self.move_joint(pose_1)
            self.current_pose = pose
            return self.move_joint(point.position_quaternion_claw)
            # self.claws_control(point.claw, 0, point)


    def send_command(self, command: str) -> str:
        """发送指令并返回响应"""
        self.sock.sendall(f"{command}\n".encode())
        response = self.sock.recv(1024).decode().strip()
        logger.info(f"Command: {command}")
        logger.info(f"Response: {response}")

        print(f"Command: {command}")
        print(f"Response: {response}")
        return response
    def get_pose(self):
        """获取机械臂当前位姿"""
        res = self.send_command("GetPose()")
        part = res.split('{')[1].split('}')[0] 
        numbers = part.split(',')
        print(part)
        return numbers
    ####11111111111111111111
    def get_angle(self):
        res = self.send_command("GetAngle()")
        part = res.split('{')[1].split('}')[0] 
        numbers = part.split(',')
        print(part)
        return numbers

    def send_modbus_command(self, modbus: str, msg):
    # 发送 ModBus 指令
        command = f"{modbus}{msg}" 
        return self.send_command(command)
    def claws_send_command(self, id, num1, num2, num3):
        command = f'SetHoldRegs({id}, {num1}, {num2}, {{{num3}}}, "U16")'
        self.send_command(command)
    def claws_read_command(self, id, num1, num2):
        command = f'GetHoldRegs({id}, {num1}, {num2}, "U16")'
        return self.send_command(command)
    def _read_claw_status(self) -> tuple:  #读取夹爪状态
        """Read the claw status via Modbus."""  
        client = ModbusTcpClient(self.ip, port=502)  # Modbus connection to robot IP  
        if not client.connect():  
            return False, "Failed to connect to Modbus server."  

        try:  
            unit_id = 2  # Device unit ID  
            read_address = 258  # Register address for claw status  
            read_response = client.read_holding_registers(read_address, count=1, unit=unit_id)  

            if not read_response.isError():  
                return True, read_response.registers[0]  # Return flag indicating success and claw status  
            else:  
                return False, str(read_response)  # Return error message if reading fails  
        except Exception as err:  
            print(f"Error reading claw status: {err}")  
            return False, err  
        finally:  
            client.close()  # Ensure the Modbus connection is closed    

    def joint_inverse_kin(self, pose: list, useJointNear=False, JointNear=[0, 0, 0, 0, 0, 0]):
        """关节运动指令封装"""
        response = self.send_command(
            f"InverseKin({','.join(str(v) for v in pose)}, "
            f"useJointNear={str(int(True))}, "
            f"jointNear={{{','.join(str(v) for v in JointNear)}}})"
        )
        res_flag = int(response.split(',')[0])
        if res_flag !=0:
             return None

        start = response.find('{')
        end = response.find('}')

        # 提取第一个大括号内的内容
        res = response[start + 1:end]
        print(res)
        # 将提取的字符串转换为列表
        res_joint = [float(num) for num in res.split(',')]
        return res_joint

    def control_movement(self, mode, value: list, a=30, v=30, wait_flag=True):
        """关节运动指令封装"""
        ALLOWED_MODES = {'joint', 'pose'}
        if mode not in ALLOWED_MODES:
            raise ValueError(f"Invalid mode: {mode}. Allowed modes are: {ALLOWED_MODES}")
        value_str = ",".join(str(v) for v in value)
        response = self.send_command(f"MovJ({mode}={{{value_str}}},a={a},v={v})")
        return response
        # response = self.send_command(f"MovJ({mode}={{{value_str}}},a={a},v={v})")
        # res_flag = int(response.split(',')[0])
        # if mode == 'pose' and res_flag not in [5, 6]:
        #     self.send_command(f"ClearError()")
        #     print('An error occurred with pose-specified motion. Attempting to use joint motion.')
        #     inverse_joint = self.joint_inverse_kin(value, useJointNear=True)
        #     # self.control_movement(mode = 'joint', value = inverse_joint)
        #     inverse_joint_str = ",".join(str(v) for v in inverse_joint)
        #     response_joint = self.send_command(f"MovJ(joint={{{inverse_joint_str}}},a={a},v={v})")
        #     res_flag = int(response_joint.split(',')[0])
        # if wait_flag:
        #     self.wait_and_prompt()
        # if res_flag == 0:
        #     return 'Success'
        # else:
        #     return res_flag

    def claws_control(self, status, id, point = None):  # (1 开 0 关)
        if point != None:
            point.claw = status
        if status: # 打开机械臂爪子
            self.claws_send_command(id, 258, 1, 0)
            self.claws_send_command(id, 259, 1, 1)
            self.claws_send_command(id, 264, 1, 1)
            self.claws_send_command(0, 258, 1, 0)
            time.sleep(1)
        else: # 关闭机械臂爪子
            self.claws_send_command(id, 258, 1, 1)
            self.claws_send_command(id, 259, 1, 0)
            self.claws_send_command(id, 264, 1, 1)
            self.claws_send_command(0, 258, 1, 1) 
            time.sleep(1)


    def move_l_pose(self,pose:str,a=30,v=30):
        logger.info(f"move_l_pose :{pose}")
        response = self.send_command(f"MovL(pose={pose},a={a},v={v})")
        res_flag = int(response.split(',')[0])
        if res_flag == 0:
            return 'Success'
        else:
            return res_flag


    def move_joint_pose(self,pose:str,a=30,v=30):
        logger.info(f"move_joint_pose :{pose}")
        response = self.send_command(f"MovJ(pose={pose},a={a},v={v})")
        res_flag = int(response.split(',')[0])
        if res_flag == 0:
            return 'Success'
        else:
            return res_flag
            
    ##########1111111111
    def move_joint(self, pose: str, a=30, v=30):
        """关节运动指令封装"""
        print("11111111111111111111111111111")
        # return self.send_command(f"MovJ(pose={pose},a={a},v={v})")
#        response = self.send_command(f"MovJ(pose={pose},a={a},v={v})")
        
#       res_flag = int(response.split(',')[0])
#
        res_flag =1
        if res_flag != 0:
            self.clear_error()
            logger.error('An error occurred with pose-specified motion. Attempting to use joint motion.')
            current_joint = self.get_angle()

            value = [float(v) for v in pose.strip('{}').split(',')]
            # inverse_joint= self.joint_inverse_kin(value, useJointNear = True,JointNear=current_joint)



            max_attempts = 5
            for attempt in range(max_attempts):
                if attempt == 0:
                    inverse_joint = self.joint_inverse_kin(value, useJointNear=True, JointNear=current_joint)
                else:
                    target_near = [float(j) + random.uniform(-5, 5) for j in value]  # 扰动范围可以根据实际情况调整
                    inverse_joint = self.joint_inverse_kin(target_near, useJointNear=True, JointNear=current_joint)
                    logger.warning(f"Attempt {attempt+1}: Using perturbed target pose: {target_near}")

                if inverse_joint is None:
                    continue




                inverse_joint_str = ",".join(str(v) for v in inverse_joint)
                response_joint = self.send_command(f"MovJ(joint={{{inverse_joint_str}}},a={a},v={v})")
                res_flag = int(response_joint.split(',')[0])
                if res_flag ==0:
                    return 'Success'
                else:
                    logger.error(f"attempt {attempt+1} failed with error code {res_flag}")
                    self.clear_error()

            
            
            # # self.control_movement(mode = 'joint', value = inverse_joint)
            # inverse_joint_str = ",".join(str(v) for v in inverse_joint)
            # response_joint = self.send_command(f"MovJ(joint={{{inverse_joint_str}}},a={a},v={v})")
            # res_flag = int(response_joint.split(',')[0])
        if res_flag == 0:
            return 'Success'
        else:
            return res_flag
    def clear_error(self):
        """清除当前警报"""
        command = f"ClearError()"
        self.send_command(command)
    def interrupt_close(self):
        """中断错误"""
        for index in range(4):
            self.send_modbus_command('Modbus', f'close({index})')
            self.send_modbus_command('Modbus', f'close({index})')
    @property
    def status(self) -> int:
        """获取机械臂状态"""
        response = self.send_command("RobotMode()")
        return int(response.split(',')[1][1])

    def switch_drag(self, status: bool):

        """更改拖拽模式"""
        command = f"StartDrag()" if status else "StopDrag()"
        self.send_command(command)

    def wait_and_prompt(self, replay = True, point = None):
        while self.status not in [5, 6]:
            time.sleep(0.1)
        if replay:
            user_input = input("机械臂空闲，输入''继续下一动作：")
            while user_input.lower() != '':
                print("输入无效，请输入''确认继续！")
                user_input = input("机械臂空闲，输入''继续下一动作：")

    def wait_and_control(self, replay = True, point = None):
        while self.status not in [5, 6]:
            time.sleep(0.1)
        if replay:
            user_input = input("机械臂空闲，输入''继续下一动作：")
            while user_input.lower() != 'q' or user_input.lower() != 'e':
                print("输入无效，请输入'q' or 'e' 确认继续！")
                if user_input.lower()=='q':
                    return True
                elif user_input.lower()=='e':
                    return False
                user_input = input("机械臂空闲，输入''继续下一动作：")

    def replay_motion_trajectory(self, modbus, timestamp, replay = True):
        trajectory_points = []
        cnt = 0
        print("检测到轨迹文件，输入''确认执行复现，其他输入取消：")
        if replay:
            user_choice = input().lower()
        
            if user_choice != '':
                print("取消轨迹复现，继续等待'a'键...")
                return 0
        def find_latest_timestamp_folder(folder_path):
            dir_list = os.listdir(folder_path)
            timestamp_folders = []
            pattern = r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_l_wbl$'# 匹配时间戳格式
            for name in dir_list:
                if os.path.isdir(os.path.join(folder_path, name)) and re.match(pattern, name):
                    timestamp_folders.append(name)
            if not timestamp_folders:
                return None
            # 按时间戳排序，取最新文件夹
            timestamp_folders = sorted(
                timestamp_folders,
                key=lambda x: datetime.strptime(x[:19], "%Y-%m-%d_%H-%M-%S"),
                reverse=True
            )
            latest_folder = timestamp_folders[0]
            return latest_folder
        try:
            # folder_path = timestamp
            folder_path = find_latest_timestamp_folder('data/left_wbl/')
            with open(os.path.join('data/left_wbl/', folder_path, 'pose.txt'), 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split(' ')
                if len(parts) >=7:  # 至少包含时间戳+6个坐标参数
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        rx = float(parts[4])
                        ry = float(parts[5])
                        rz = float(parts[6])
                        trajectory_points.append({
                            'x':x, 'y':y, 'z':z,
                            'rx':rx, 'ry':ry, 'rz':rz
                        })
                    except ValueError:
                        print(f"警告：无效数据行：{line}")
            if len(trajectory_points) >=1:
                print(f"开始轨迹复现：{os.path.join('data/left_wbl/', folder_path, 'pose.txt')}")
                for point in trajectory_points:
                    point_str = f"{{{point['x']:.4f},{point['y']:.4f},{point['z']:.4f},{point['rx']},{point['ry']},{point['rz']}}}"
                    self.move_joint(point_str)
                    self.wait_and_prompt(replay = False)
                    if cnt == 1:
                        self.wait_and_prompt(replay = True)
                        self.claws_control(0, modbus)
                        
                    elif cnt == 4:
                        self.claws_control(1, modbus)
                        self.wait_and_prompt(replay = False)
                    cnt += 1
                self.switch_drag(True)
                print("轨迹复现完成！")
            else:
                print("轨迹点不足，未执行复现")
        except FileNotFoundError:
            print("轨迹文件未找到，无法复现轨迹")

def open_gripper():
    server = Server('192.168.201.1', 29999, '192.168.110.235', 12345)
    server.sock.connect((server.ip, server.port))
    bot = DobotController(server.sock)
    server.init_com(bot)
    bot._initialize(server)

    # 程序运行后**立刻**打开夹爪
    bot.claws_control(1, server.modbusRTU)
    print("夹爪已打开！")
def main():
    def generate_position():
        """生成随机位置数据（根据实际需求调整范围）"""
        return {
            'x': round(random.uniform(-1000, 1000), 4),
            'y': round(random.uniform(-1000, 1000), 4),
            'z': round(random.uniform(0, 500), 4),
            'rx': random.randint(0, 360),
            'ry': random.randint(-180, 180),
            'rz': random.randint(-180, 180)
        }
    # 创建一个 TCP/IP 套接字
    server = Server('192.168.5.100', 29999, '192.168.110.235', 12345)
    # 连接到 DoBot 机械臂的 Dashboard 端口 (29999)
    server.sock.connect((server.ip, server.port))
    bot = DobotController(server.sock)
    server.init_com(bot)
    # 初始化机器人
    bot._initialize(server)
    # point_list = [
    # Point(
    #     name=f"point_{i:05d}",  # 格式化命名（如point_00001）
    #     position=generate_position()
    # )
    # for i in range(10)
# ]
    # def on_key_press(key, server):
    #     if key == keyboard.KeyCode.from_char('q'):
    #         server.signal['replay'] = True
    #     elif key == keyboard.KeyCode.from_char('e'):
    #         server.signal['claw_close']  = True
    #     elif key == keyboard.KeyCode.from_char('r'):
    #         server.signal['claw_open']  = True
    #     elif key == keyboard.KeyCode.from_char('o'):
    #         server.signal['set_drag']  = True
    #     elif key == keyboard.KeyCode.from_char('p'):
    #         server.signal['reset_drag']  = True
    #     elif key == keyboard.KeyCode.from_char('w'):
    #         server.signal['play'] = True

    # def make_on_key_press(server):  
    #     def on_key_press(key):  
    #         if key == keyboard.KeyCode.from_char('q'):  
    #             server.signal['replay'] = True  
    #         elif key == keyboard.KeyCode.from_char('e'):  
    #             server.signal['claw_close'] = True  
    #         elif key == keyboard.KeyCode.from_char('r'):  
    #             server.signal['claw_open'] = True  
    #         elif key == keyboard.KeyCode.from_char('o'):  
    #             server.signal['set_drag'] = True  
    #         elif key == keyboard.KeyCode.from_char('p'):  
    #             server.signal['reset_drag'] = True  
    #         elif key == keyboard.KeyCode.from_char('w'):  
    #             server.signal['play'] = True  
    #     return on_key_press
    # listener = keyboard.Listener(on_press=make_on_key_press(server))
    # listener.start()
    listen_app = threading.Thread(target = server.start_server, args = ())
    listen_app.start()
    print("------------")
    
    # bot._read_claw_status()  0,{0,1},GetHoldRegs(4, 258, 2, "U16");
    print(bot.claws_read_command(server.modbusRTU, 258, 2))
    time.sleep(1)
    bot.claws_control(0, server.modbusRTU)
    time.sleep(1)
    print(bot.claws_read_command(server.modbusRTU, 258, 1))
    time.sleep(1)
    bot.claws_control(1, server.modbusRTU)
    time.sleep(1)
    print(bot.claws_read_command(server.modbusRTU, 258, 1))
    print("------------")
    try:
        while True:

            print("按下 'q' 键轨迹复现, 'e'键夹爪关闭, 'r'键夹爪打开...")

            # 等待用户按下 'a' 键
            while not any(server.signal.values()):
                pass
            print("开始机械臂运动...")
            
            # # 依次发送关节运动指令
            dt = datetime.now()
            micro = dt.microsecond // 1000
            timestamp_start = dt.strftime(f"%Y-%m-%d_%H-%M-%S-{micro:03d}_r_wbl")

            if server.signal['replay']:
                bot.replay_motion_trajectory(server.modbusRTU, timestamp_start)
            elif server.signal['claw_close']:
                bot.claws_control(0, server.modbusRTU)
                bot.wait_and_prompt()
            elif server.signal['claw_open']:
                bot.claws_control(1, server.modbusRTU)
                bot.wait_and_prompt()
            elif server.signal['set_drag']:
                # bot.switch_drag(True)
                bot.get_pose()
            elif server.signal['reset_drag']:
                bot.switch_drag(False)
            elif server.signal['play']:
                for point in point_list:
                    joint_positions = point.position
                    joint_angles = f"{{{joint_positions['x']:.4f},{joint_positions['y']:.4f},{joint_positions['z']:.4f},{joint_positions['rx']},{joint_positions['ry']},{joint_positions['rz']}}}"
                    bot.move_joint(joint_angles)
                    bot.wait_and_prompt()

            print("机械臂运动完成。")
            for key in server.signal:
                server.signal[key] = False
            
    finally:
        # 关闭套接字连接
        bot.interrupt_close()
        server.sock.close()
        server.app.close()
        # listener.stop()

if __name__ == "__main__":
    main()
