import random
import socket
import re
import time
import os
import threading 
from datetime import datetime
# from pynput import keyboard
# from scipy.spatial.transform import Rotation as R
# import transforms3d.quaternions as quat 
import numpy as np
import sys
import struct
import traceback
import glob
import logging
from logging.handlers import RotatingFileHandler
import atexit
import math

log_directory = "logs/"

if not os.path.exists(log_directory):
    try:
        os.makedirs(log_directory)
    except Exception as e:
        raise

logger = logging.getLogger("dobot_log")
logger.setLevel(logging.INFO)
unique = os.getpid()

log_filename = f'dobot_{time.strftime("%Y-%m-%d_%H-%M-%S")}_pid{unique}.log'
# log_filename = f'dobot_{time.time()}.log'
file_handler = RotatingFileHandler(os.path.join(log_directory, log_filename),maxBytes= 1024*1024*200,backupCount=10)
# console_handler = logging.StreamHandler()  # 添加终端log输出

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
# console_handler.setFormatter(formatter)
# console_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
# logger.addHandler(console_handler)

def clean_logs(log_dir, max_files=10):
    
    log_files = glob.glob(os.path.join(log_dir, "*.log*"))
    
    log_files.sort(key=os.path.getmtime)
    if len(log_files) > max_files:
        for file_to_delete in log_files[:-max_files]:
            try:
                os.remove(file_to_delete)
                # logger.info(f"Deleted old log file: {file_to_delete}")
            except Exception as e:
                logger.error(f"Failed to delete old log file {file_to_delete}: {e}")

clean_logs(log_directory)


class Server():
    def __init__(self, host_control, port_control, host_feedback, port_feedback, modbus: bool = False, control: bool = False, feedback: bool = False):
        atexit.register(self.interrupt_close)
        self.control_lock = threading.Lock() 
        self.host_control = host_control
        self.port_control = port_control
        self.sock_control = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock_control.settimeout(10.0)
        self.host_feedback = host_feedback
        self.port_feedback = port_feedback
        self.sock_feedback = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock_feedback.settimeout(10.0)
        self.control = control
        self.feedback = feedback
        self.modbus = modbus
        self.modbus_flag = False
        self.control_flag = False
        self.feedback_flag = False
        # self.tcp = None
        # self.host = host
        # self.app_port = app_port
        self.baudrate = 115200
        self.modbus_id = None
        self.modbusRTU_id = None
        self.timestamp = None
        self.signal = {'replay':False,'claw_open':False,'claw_close':False, 'set_drag':False, 'reset_drag':False}
        self.ROBOT_ERRORCODE_DESCRIPTION = {
            -1: "命令执行失败",
            -2: "机器人处于报警状态",
            -3: "机器人处于急停状态",
            -4: "机械臂处于下电状态",
            -5: "机械臂处于脚本运行/暂停状态",
            -10000: "命令错误，下发的命令不存在",
            -20000: "参数数量错误",
        }

        # self.init_socket_connect()
        if self.control:
            self.control_init()
        if self.feedback:
            self.infor_init()
        if self.modbus:
            # self.init_socket_connect()
            if not self.control_flag:
                self.control_init() 
            self.init_modbus()
    # def init_socket_connect(self):
    #     try:
    #         self.sock_control.connect((self.host_control, self.port_control))
    #         # self.sock_feedback.connect((self.host_feedback, self.port_feedback))
    #         print("Connected to the control servers successfully.")
    #     except Exception as e:
    #         print(f"Failed to connect to the control servers: {e}")
    #         # traceback.print_exc()
    #         exit()
    def control_init(self):
        try:
            self.sock_control.connect((self.host_control, self.port_control))
            logger.info("Connected to the control servers successfully.")
            self.control_flag = True
        except Exception as e:
            logger.error(f"Failed to connect to the control servers: {e}")
            # traceback.print_exc()
            exit()
    def infor_init(self):
        try:
            self.sock_feedback.connect((self.host_feedback, self.port_feedback))
            logger.info("Connected to the feedback servers successfully.")
            self.feedback_flag = True
        except Exception as e:
            logger.error(f"Failed to connect to the feedback servers: {e}")
            # traceback.print_exc()
            exit()
    def init_modbus(self):
        msg_modbus = f'Create("{self.host_control}", 502, 2)'
        msg_modbusrtu = f'Create(1, {self.baudrate}, "N", 8, 1)'
        modbus_response = self.send_modbus_command('Modbus', msg_modbus)
        modbusRTU_response = self.send_modbus_command('ModbusRTU', msg_modbusrtu)
        
        res_flag_modbus = int(modbus_response.split(',')[0])
        res_flag_modbusRTU = int(modbusRTU_response.split(',')[0])
        if res_flag_modbus !=0 or res_flag_modbusRTU !=0:
            for id in range(5):
                self.send_modbus_command('Modbus', f'Close({id})')
            self.modbus_id = self._parse_id(self.send_modbus_command('Modbus', msg_modbus))
            self.modbusRTU_id = self._parse_id(self.send_modbus_command('ModbusRTU', msg_modbusrtu))
            self.modbus_flag = True
            return
        self.modbus_id = self._parse_id(modbus_response)
        self.modbusRTU_id = self._parse_id(modbusRTU_response)
        self.modbus_flag = True

    def _parse_id(self, response):
        match = re.search(r'\{(.+?)\}', response)
        if match:
            try:
                numbers_str = match.group(1)
                return int(numbers_str)
            except (ValueError, TypeError):
                return None

    def send_command(self, command: str) -> str:
        """发送指令并返回响应"""
        with self.control_lock:  
            try:
                self.sock_control.sendall(f"{command}\n".encode())
                response = self.sock_control.recv(1024).decode().strip()
                logger.info(f"Command: {command}")
                logger.info(f"Response: {response}")
                res_flag = int(response.split(',')[0])
                if 0 != res_flag:
                    if res_flag in self.ROBOT_ERRORCODE_DESCRIPTION:
                        logger.error(f"response error,error code description is:{self.ROBOT_ERRORCODE_DESCRIPTION[res_flag]}")
                    else:
                        logger.error(f"response error,error code undefined")
                return response
            except socket.error as e:
                logger.error(f"Socket error: {e}")
                raise
    def send_modbus_command(self, modbus: str, msg):
    # 发送 ModBus 指令
        command = f"{modbus}{msg}" 
        return self.send_command(command)
    def interrupt_close(self):
        if self.modbus_flag:
            self.send_modbus_command('Modbus', f'Close({self.modbus_id})')
            self.send_modbus_command('Modbus', f'Close({self.modbusRTU_id})')
            self.modbus_flag = False
        if self.control_flag:
            self.sock_control.close()
            self.control_flag =False
        if self.feedback_flag:
            self.sock_feedback.close()
            self.feedback_flag = False
    # def start_server(self):
    #     """启动服务器"""
    #     self.app = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #     self.app.bind((self.host, self.app_port))
    #     self.app.listen(5)  # 最大等待连接数为5
    #     print(f"Server listening on {self.host}:{self.app_port}")

    #     try:
    #         while True:
    #             self.tcp, addr = self.app.accept()
    #             print(f"Accepted connection from {addr[0]}:{addr[1]}")
    #             # 创建新线程处理客户端
    #             client_handler = threading.Thread(
    #                 target=self.handle_client,
    #                 args=(self.tcp, )
    #             )
    #             client_handler.daemon = True
    #             client_handler.start()
    #     except KeyboardInterrupt:
    #         print("Stopping server...")
    #         self.app.close()
    #         self.tcp.close()
    # def handle_client(self):
    #     """处理客户端的通信"""
    #     # global replay_motion
    #     # global claw_open
    #     # global claw_close
    #     try:
    #         while True:
    #             data = self.sock.recv(1024).decode().strip()
    #             if not data:
    #                 # 客户端断开连接
    #                 print("Client disconnected")
    #                 break
    #             print(f"Received data: {data}")
    #             if data == 'replay':
    #                 self.signal['replay'] = True
    #             elif data == 'open':
    #                 self.signal['claw_open'] = True
    #             elif data == 'close':
    #                 self.signal['claw_close'] = True
    #             elif data == 'set':
    #                 self.signal['set_drag'] = True
    #             elif data == 'reset':
    #                 self.signal['reset_drag'] = True
    #             self.sock.send("Message received".encode())
    #     except socket.error as e:
    #         print(f"Socket error: {e}")
    #     finally:
    #         # 关闭客户端 socket
    #         self.sock.close()
# class Point():
#     def __init__(self,  position: list, claw, quaternion: list = None, euler: list = None,  name=None):
#         self.name = name  
#         self.position = position
#         # self.quaternion = quaternion
#         # self.euler = None
#         self.timestamp = None
#         self.position_quaternion_claw = None
#         self.claw = claw
#         self.quaternion = quaternion
#         self.euler = euler
        
#         self.__position_to_string()
#         self.__euler_to_string()
#         self.__quaternion_to_euler()
#         self.__get_position_and_quaternion()
#     def get_timestamp(self):
#         dt = datetime.now()
#         micro = dt.microsecond // 1000
#         self.timestamp = dt.strftime(f"%Y-%m-%d_%H-%M-%S-{micro:03d}")
#         return self.timestamp
#     def __position_to_string(self):
#         if self.position is None:
#             return None
#         # self.position_str = f"{{{self.position['x']:.4f},{self.position['y']:.4f},{self.position['z']:.4f},\
#         #     {self.position['rx']},{self.position['ry']},{self.position['rz']}}}"
#         self.position = f"{{{self.position[0]:.4f},{self.position[1]:.4f},{self.position[2]:.4f},"
        
#         # return self.position_str
#     def __euler_to_string(self):
#         if self.euler is None:
#             return None
#         # self.position_str = f"{{{self.position['x']:.4f},{self.position['y']:.4f},{self.position['z']:.4f},\
#         #     {self.position['rx']},{self.position['ry']},{self.position['rz']}}}"
#         self.euler = f"{self.euler[0]:.4f},{self.euler[1]:.4f},{self.euler[2]:.4f}}}"
#     def __quaternion_to_euler(self):
#         if self.quaternion and not self.euler:
#             # 将列表转换为 numpy 数组（如果需要）
#             self.quaternion = np.array(self.quaternion)

#             # 确保四元数是单位四元数
#             self.quaternion = self.quaternion / np.linalg.norm(self.quaternion)

#             # 使用 transforms3d 进行四元数到欧拉角的转换
#             euler_angles_rad = quat.quat2euler(self.quaternion)  # 返回的是弧度

#             # 转换为欧拉角（角度）
#             euler_angles_deg = np.degrees(euler_angles_rad)
#             self.euler = f"{euler_angles_deg[0]:.4f},{euler_angles_deg[1]:.4f},{euler_angles_deg[2]:.4f}}}"

#             return euler_angles_deg
    # 四元数转欧拉角
    # def __quaternion_to_euler(self):
    #     # scipy的实现
    #     if self.quaternion and not self.euler:
    #         # 将列表转换为 numpy 数组（如果需要）
    #         self.quaternion = np.array(self.quaternion)

    #         # 确保四元数是单位四元数
    #         self.quaternion = self.quaternion / np.linalg.norm(self.quaternion)

    #         # 创建 Rotation 对象
    #         rotation = R.from_quat(self.quaternion)

    #         # 转换为欧拉角（弧度）
    #         euler_angles_rad = rotation.as_euler('xyz')

    #         # 转换为欧拉角（角度）
    #         euler_angles_deg = np.degrees(euler_angles_rad)
    #         self.euler = f"{euler_angles_deg[0]:.4f},{euler_angles_deg[1]:.4f},{euler_angles_deg[2]:.4f}}}"

    #         return euler_angles_deg

    # def __get_position_and_quaternion(self):
    #     self.position_quaternion_claw = str(self.position) + str(self.euler)
class DobotController():
    def __init__(self, server):
        self.sock_control = server.sock_control
        self.sock_feedback = server.sock_feedback
        self.server = server
        self.modbus_id = server.modbus_id
        self.modbusRTU_id = server.modbusRTU_id
        # self._initialize()
        # self.ROBOT_MODE_MAP = {
        #     "ROBOT_MODE_INIT": 1,
        #     "ROBOT_MODE_BRAKE_OPEN": 2,
        #     "ROBOT_MODE_POWEROFF": 3,
        #     "ROBOT_MODE_DISABLED": 4,
        #     "ROBOT_MODE_ENABLED": 5,
        #     "ROBOT_MODE_BACKDRIVE": 6,
        #     "ROBOT_MODE_RUNNING": 7,
        #     "ROBOT_MODE_PAUSED": 8,
        #     "ROBOT_MODE_ERROR": 9
        # }
        self.ROBOT_MODE_DESCRIPTION = {
            1: "初始化状态",
            2: "有任意关节的抱闸松开",
            3: "机械臂下电状态",
            4: "未使能（无抱闸松开）",
            5: "使能状态",
            6: "拖拽状态",
            7: "运行状态",
            8: "暂停状态",
            9: "错误状态",
            10: "pause"
        }

        self.clear_error()
        self.robot_continue()

    def _initialize(self):
        self.server.send_command("PowerOn()")
        time.sleep(1)
        self.server.send_command("EnableRobot()")
        self.server.send_command("ClearError()")
        # self.modbus_id = self.server.modbus_id
        # self.modbusRTU_id = self.server.modbusRTU_id
    # def control_init(self):
    #     try:
    #         self.sock_control.connect((self.server.host_control, self.server.port_control))
    #         print("Connected to the control servers successfully.")
    #     except Exception as e:
    #         print(f"Failed to connect to the control servers: {e}")
    #         # traceback.print_exc()
    #         exit()
    # def infor_init(self):
    #     try:
    #         self.sock_feedback.connect((self.server.host_feedback, self.server.port_feedback))
    #         print("Connected to the feedback servers successfully.")
    #     except Exception as e:
    #         print(f"Failed to connect to the feedback servers: {e}")
    #         # traceback.print_exc()
    #         exit()
    # def point_control(self, point = None):
    #     if point != None:
    #         self.move_joint(point.position_quaternion_claw)
    #         # self.claws_control(point.claw, 0, point)
    def claws_send_command(self, id, addr, count, value, value_type = "U16"):
        if type(value) == list: 
            command = f'SetHoldRegs({id}, {addr}, {count}, {{{",".join([str(v) for v in value])}}}, {value_type})'
        else:
            command = f'SetHoldRegs({id}, {addr}, {count}, {value}, {value_type})'
        self.server.send_command(command)
    def claws_read_command(self, id, addr, count, value_type = "U16"):
        command = f'GetHoldRegs({id}, {addr}, {count}, {value_type})'
        return self.server.send_command(command)
    def changingtek_open_degree(self, set_degree, id, point = None, step = False):
        #set_degree表示夹爪张开角度 100为完全打开
        if point != None:
            point.claw = set_degree
        if set_degree < 0:
            set_degree = 0
        if set_degree > 100:
            set_degree = 100
        control_value = int(9000 - set_degree * 9000 / 100) # 传入角度线性转换成控制参数
        self.claws_send_command(id, 258, 1, [0])
        self.claws_send_command(id, 259, 1, [control_value])
        self.claws_send_command(id, 264, 1, [1])
        time.sleep(1)
        if step:
            self.wait_and_prompt()
    @property
    def get_pose(self) ->tuple:
        response = self.server.send_command(f"GetPose()")
        if response.startswith("0,"):  # 检查ErrorID=0（成功）
            try:
                start_idx = response.find("{") + 1
                end_idx = response.find("}")
                pose_str = response[start_idx:end_idx]
                pose = tuple(map(float, pose_str.split(",")))
                return pose
            except (ValueError, IndexError):
                logger.error(f"解析GetPose响应失败: {response}")
        else:
            logger.error("clear error")
            self.clear_error()
            self.robot_continue()
        return None
    @property
    def get_angle(self) ->tuple:
        response = self.server.send_command(f"GetAngle()")
        if response.startswith("0,"):  # 检查ErrorID=0（成功）
            try:
                start_idx = response.find("{") + 1
                end_idx = response.find("}")
                joint_str = response[start_idx:end_idx]
                joint = tuple(map(float, joint_str.split(",")))
                return joint
            except (ValueError, IndexError):
                logger.error(f"解析GetAngle响应失败: {response}")
        return None

    def move_joint(self, pose: str, a=30, v=30):
        """关节运动指令封装"""
        self.server.send_command(f"MovJ(pose={pose},a={a},v={v})")
        
    def control_move_arc(self,mode,pose_mid : list,pose_target : list,a = 30, v = 30, wait_flag=True):
        pose_mid_str = ",".join(str(v) for v in pose_mid)
        pose_target_str = ",".join(str(v) for v in pose_target)

        response = self.server.send_command(f"Arc({mode}={{{pose_mid_str}}},{mode}={{{pose_target_str}}},a={a},v={v})")
        res_flag = int(response.split(',')[0])
        if wait_flag:
            self.wait_and_prompt(mode,pose_target)
        if res_flag == 0:
            return 'Success'
        else:
            return res_flag


    def control_servop(self,mode,pose:list):
        if mode != 'pose':
            logger.error(f"control servop  invalid mode: {mode}")
            return
        pose_str = ",".join(str(v) for v in pose)
        # response = self.server.send_command(f"ServoP({pose_str},t=0.5,aheadtime = 70,gain = 250)")
        response = self.server.send_command(f"ServoP({pose_str},t=0.5)")

        res_flag = int(response.split(',')[0])
        if res_flag == 0:
            return 'Success'
        else:
            return res_flag
    def control_servoj(self,mode,joint:list):
        if mode != 'joint':
            logger.error(f"control servoj  invalid mode: {mode}")
            return
        joint_str = ",".join(str(v) for v in joint)
        # response = self.server.send_command(f"ServoP({pose_str},t=0.5,aheadtime = 70,gain = 250)")
        response = self.server.send_command(f"ServoJ({joint_str},t=0.5)")

        res_flag = int(response.split(',')[0])
        if res_flag == 0:
            return 'Success'
        else:
            return res_flag          
            
            
        
    def control_movement(self, mode, value: list, a = 10, v = 80, wait_flag = True):
        cur_pose = self.get_pose
        """关节运动指令封装"""
        ALLOWED_MODES = {'joint', 'pose'}
        if mode not in ALLOWED_MODES:
            raise ValueError(f"Invalid mode: {mode}. Allowed modes are: {ALLOWED_MODES}")
        value_str = ",".join(str(v) for v in value)
        response = self.server.send_command(f"MovJ({mode}={{{value_str}}},a={a},v={v})")
        res_flag = int(response.split(',')[0])
        if(0 != res_flag):
            logger.error("an error occured with movJ")
        if mode == 'pose' and res_flag == 9:
            self.clear_error()
            logger.warning('An error occurred with pose-specified motion. Attempting to use joint motion.')
            inverse_joint= self.joint_inverse_kin(value, useJointNear = True)
            # self.control_movement(mode = 'joint', value = inverse_joint)
            inverse_joint_str = ",".join(str(v) for v in inverse_joint)
            response_joint = self.server.send_command(f"MovJ(joint={{{inverse_joint_str}}},a={a},v={v})")
            res_flag = int(response_joint.split(',')[0])
            if wait_flag:
                self.wait_and_prompt('joint',response_joint)
            if res_flag == 0:
                return 'Success'
            else:
                return res_flag

        if wait_flag:
            self.wait_and_prompt(mode,value)
        if res_flag == 0:
            return 'Success'
        else:
            return res_flag
    def set_bot_speed(self, speed_ratio: int):
        response = self.server.send_command(f"SpeedFactor({speed_ratio})")
        res_flag = int(response.split(',')[0])
        return res_flag
    @property
    def status(self) -> int:
        """获取机械臂状态"""
        response = self.server.send_command("RobotMode()")
        robotmode = int(response.split(',')[1][1])
        if robotmode not in [5, 6]:
            logger.info(f"机械臂未空闲，当前状态：{self.ROBOT_MODE_DESCRIPTION[robotmode]}")


        return robotmode
    
    @property
    def changingtek_get_degree(self) -> int:
        """获取夹爪状态（夹爪张开角度）"""
        # response = self.claws_read_command(self.modbusRTU_id, 258, 2)
        response = self.claws_read_command(self.modbusRTU_id, 0x60D, 2) # 直接读取夹爪编码器反馈值
        match = re.search(r'{(.*?)}', response)
        if match:
            content = match.group(1)
        status = [int(value) for value in (content.split(','))]
        set_degree = (9000 - ((status[0] << 16) + status[1]))/ 9000 * 100
        logger.info(f"夹爪张开程度{set_degree}")
        return set_degree
    @property
    def get_end_pose(self) -> tuple:
        response = self.sock_feedback.recv(1440)
        if response:
            pose = struct.unpack('<6d', response[624:672])
            logger.info(f"get_end_pose:{pose}")
        return pose
    @property
    def get_joint_degree(self) -> tuple:
        response = self.sock_feedback.recv(1440)
        if response:
            deg = struct.unpack('<6d', response[432:480])
            logger.info(f"get_jont_degree:{deg}")
        return deg 
    def switch_drag(self, status: bool, step = False):

        """更改拖拽模式"""
        command = f"StartDrag()" if status else "StopDrag()"
        self.server.send_command(command)
    def clear_error(self):
        """清除当前警报"""
        command = f"ClearError()"
        self.server.send_command(command)
    def robot_continue(self):
        """清除当前警报"""
        command = f"Continue()"
        self.server.send_command(command)

    def joint_inverse_kin(self, pose: list, useJointNear = False, JointNear = [0, 0, 0, 0, 0, 0]):
        """关节运动指令封装"""
        response = self.server.send_command(
                f"InverseKin({','.join(str(v) for v in pose)}, "
                f"useJointNear={str(int(True))}, "
                f"jointNear={{{','.join(str(v) for v in JointNear)}}})"
                )
        start = response.find('{')
        end = response.find('}')

        # 提取第一个大括号内的内容
        res = response[start + 1:end]

        # 将提取的字符串转换为列表
        res_joint = [float(num) for num in res.split(',')]
        return res_joint
    def wait_and_prompt(self,mode = None,target_position =None):
        start_time = time.time()
        if(target_position == None or mode ==None):
            try:
                status = 1
                while(status):
                    end_time = time.time()
                    if end_time - start_time >20:
                        break
                    status = self.status
                    time.sleep(0.1)
                    if status not in [5, 6]:
                        if status == 9:
                            raise RuntimeError("机械臂处于错误状态")
                        pass
                    elif status == 5:
                        break
                    # time.sleep(0.1)
            except BaseException as e:
                logger.error("错误码:", status)
                raise e
        else:
            if mode == 'pose':
                try:
                    while(True):
                        end_time = time.time()
                        if end_time - start_time >20:
                            break
                        if self.status == 9:
                            raise RuntimeError("机械臂处于错误状态")
                        current_pose = self.get_pose
                        if None == current_pose:
                            continue
                        pos_error = math.dist(current_pose[:3], target_position[:3])
                        rot_error = max(abs(c - t) for c, t in zip(current_pose[3:], target_position[3:]))
                        if(pos_error<2 and rot_error <1):
                            break
                        time.sleep(0.2)
                        
                except BaseException as e:
                    raise e
            elif mode =='joint':   
                try:
                    while(True):
                        end_time = time.time()
                        if end_time - start_time >20:
                            break
                        if self.status == 9:
                            raise RuntimeError("机械臂处于错误状态")
                        current_joint = self.get_angle
                        if None == current_joint:
                            continue
                        angle_error = max(abs(c - t) for c, t in zip(current_joint[:6], target_position[:6]))
                        if(angle_error <1):
                            break
                        time.sleep(0.1)
                        
                except BaseException as e:
                    raise e    
    @property
    def get_claws_torque(self):
        response = self.claws_read_command(self.modbusRTU_id, 0x60C, 1)
        start = response.find('{')
        end = response.find('}')
        res_torque = response[start + 1:end]

        return int(res_torque)
        # if step:
        #     user_input = input("机械臂空闲，输入''继续下一动作：")
        #     while user_input.lower() != '':
        #         print("输入无效，请输入''确认继续！")
        #         user_input = input("机械臂空闲，输入''继续下一动作：")
    # def replay_motion_trajectory(self, modbus, timestamp, replay = True):
    #     trajectory_points = []
    #     cnt = 0
    #     print("检测到轨迹文件，输入''确认执行复现，其他输入取消：")
    #     if replay:
    #         user_choice = input().lower()
            
    #         if user_choice != '':
    #             print("取消轨迹复现，继续等待'a'键...")
    #             return 0
    #     def find_latest_timestamp_folder(folder_path):
    #         dir_list = os.listdir(folder_path)
    #         timestamp_folders = []
    #         pattern = r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_l_wbl$'# 匹配时间戳格式
    #         for name in dir_list:
    #             if os.path.isdir(os.path.join(folder_path, name)) and re.match(pattern, name):
    #                 timestamp_folders.append(name)
    #         if not timestamp_folders:
    #             return None
    #         # 按时间戳排序，取最新文件夹
    #         timestamp_folders = sorted(
    #             timestamp_folders,
    #             key=lambda x: datetime.strptime(x[:19], "%Y-%m-%d_%H-%M-%S"),
    #             reverse=True
    #         )
    #         latest_folder = timestamp_folders[0]
    #         return latest_folder
    #     try:
    #         # folder_path = timestamp
    #         pose_path = find_latest_timestamp_folder('data/left_wbl/')
    #         with open(os.path.join('data/left_wbl/', pose_path, 'pose.txt'), 'r') as f:
    #             lines = f.readlines()
    #         for line in lines:
    #             parts = line.strip().split(' ')
    #             if len(parts) >=7:  # 至少包含时间戳+6个坐标参数
    #                 try:
    #                     x = float(parts[1])
    #                     y = float(parts[2])
    #                     z = float(parts[3])
    #                     rx = float(parts[4])
    #                     ry = float(parts[5])
    #                     rz = float(parts[6])
    #                     trajectory_points.append({
    #                         'x':x, 'y':y, 'z':z,
    #                         'rx':rx, 'ry':ry, 'rz':rz
    #                     })
    #                 except ValueError:
    #                     print(f"警告：无效数据行：{line}")
    #         if len(trajectory_points) >=1:
    #             print(f"开始轨迹复现：{os.path.join('data/left_wbl/', pose_path, 'pose.txt')}")
    #             for point in trajectory_points:
    #                 point_str = f"{{{point['x']:.4f},{point['y']:.4f},{point['z']:.4f},{point['rx']},{point['ry']},{point['rz']}}}"
    #                 self.move_joint(point_str)
    #                 self.wait_and_prompt(replay = False)
    #                 if cnt == 1:
    #                     self.wait_and_prompt(replay = True)
    #                     self.claws_control(0, modbus)
                        
    #                 elif cnt == 4:
    #                     self.claws_control(1, modbus)
    #                     self.wait_and_prompt(replay = False)
    #                 cnt += 1
    #             self.switch_drag(True)
    #             print("轨迹复现完成！")
    #         else:
    #             print("轨迹点不足，未执行复现")
    #     except FileNotFoundError:
    #         print("轨迹文件未找到，无法复现轨迹")
# def main():
#     # def generate_position():
#     #     """生成随机位置数据（根据实际需求调整范围）"""
#     #     return {
#     #         'x': round(random.uniform(-1000, 1000), 4),
#     #         'y': round(random.uniform(-1000, 1000), 4),
#     #         'z': round(random.uniform(0, 500), 4),
#     #         'rx': random.randint(0, 360),
#     #         'ry': random.randint(-180, 180),
#     #         'rz': random.randint(-180, 180)
#     #     }
#     def capture_key_press(server):
#         def on_key_press(key):
#             print(key)
#             if key == keyboard.KeyCode.from_char('q'):
#                 server.signal['replay'] = True
#             elif key == keyboard.KeyCode.from_char('e'):
#                 server.signal['claw_close']  = True
#             elif key == keyboard.KeyCode.from_char('r'):
#                 server.signal['claw_open']  = True
#             elif key == keyboard.KeyCode.from_char('o'):
#                 server.signal['set_drag']  = True
#             elif key == keyboard.KeyCode.from_char('p'):
#                 server.signal['reset_drag']  = True
#             elif key == keyboard.KeyCode.from_char('w'):
#                 server.signal['play'] = True
#         return on_key_press
#     try:
#     # 创建一个 TCP/IP 套接字
#         server = Server('192.168.201.1', 29999, '192.168.201.1', 30004)
#         # 连接到 DoBot 机械臂的 Dashboard 端口 (29999)
#         # server.sock_control.connect()
#         # server.sock_feedback.connect()
        
#         bot = DobotController(server)
#         bot.control_init()
#         bot.infor_init()
#         server.init_modbus()
#         # 初始化机器人
#         bot._initialize()
#         point_list = [
#         Point(
#             name=f"p0",  # 格式化命名（如point_00001）
#             position=[807.1072323847449, 111.6522978028797, 300.0],
#             euler=[ 180.0, 0.0, -174.1942787708895],
#             claw= False
#         )
#     ]
#         listener = keyboard.Listener(on_press=capture_key_press(server))
#         listener.start()
#         # listen_app = threading.Thread(target = server.start_server, args = ())
#         # listen_app.start()
#         try:
#             while True:

#                 print("按下 'q' 键轨迹复现, 'e'键夹爪关闭, 'r'键夹爪打开, 'o'键拖拽机械臂, 'p'键关闭拖拽机械臂, 'w'键机械臂移动到指定点...")

#                 # 等待用户按下 'a' 键
#                 while not any(server.signal.values()):
#                     # sys.stdin.read()
#                     pass
#                 print("开始机械臂运动...")
                
#                 # # 依次发送关节运动指令
#                 dt = datetime.now()
#                 micro = dt.microsecond // 1000
#                 timestamp_start = dt.strftime(f"%Y-%m-%d_%H-%M-%S-{micro:03d}_r_wbl")

#                 if server.signal['replay']:
#                     bot.replay_motion_trajectory(server.modbusRTU, timestamp_start)
#                 elif server.signal['claw_close']:
#                     bot.changingtek_open_degree(10, server.modbusRTU)
#                     bot.wait_and_prompt()
#                 elif server.signal['claw_open']:
#                     bot.changingtek_open_degree(90, server.modbusRTU)
#                     bot.wait_and_prompt()
#                 elif server.signal['set_drag']:
#                     bot.switch_drag(True)
#                 elif server.signal['reset_drag']:
#                     bot.switch_drag(False)
#                 elif server.signal['play']:
#                     for cnt in range(1):
#                         # joint_positions = point.position
#                         # joint_angles = f"{{{joint_positions['x']:.4f},{joint_positions['y']:.4f},{joint_positions['z']:.4f},{joint_positions['rx']},{joint_positions['ry']},{joint_positions['rz']}}}"
#                         bot.control_movement(mode = 'joint', value= [38.11945724487305, 2.9753165245056152, -98.84064483642578, 1.9852956533432007, 93.61241149902344, -257.98040771484375])
#                         # bot.control_movement('pose', [430.1399230957031, 166.5605926513672, 424.53173828125, 176.9378662109375, -4.5936079025268555, -154.02188110351562])
#                         bot.wait_and_prompt()
#                         # bot.changingtek_open_degree(cnt,server.modbusRTU)
#                         # print(bot.changingtek_get_degree)
#                         # bot.wait_and_prompt()
#                         # bot.claws_control(30,server.modbusRTU)
#                         # bot.wait_and_prompt()
#                         # print(bot.claws_status)
#                 print("机械臂运动完成。")
#                 for key in server.signal:
#                     server.signal[key] = False
#         finally:
#             listener.stop()
#     finally:
#         # 关闭套接字连接
#         server.interrupt_close()
#         # server.app.close()


# if __name__ == "__main__":
#     main()
