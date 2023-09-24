import socket
import threading
from pygame.time import Clock
import re
import numpy as np
from math import pi, sin, cos
import torch
import time
import config
from net import TransPoseNet
from articulate.math import *
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
inertial_poser = TransPoseNet(num_past_frame=60, num_future_frame=5).to(device)
running = False
start_recording = False
start_aligning = False


class IMUSet:

    # imu_host改成手机的ip地址
    # def __init__(self, imu_host='10.17.223.169', imu_port=9090, buffer_len=40):
    def __init__(self, imu_host='10.203.161.205', imu_port=9090, buffer_len=40):
        self.imu_host = imu_host
        self.imu_port = imu_port
        self.clock = Clock()
        self._imu_socket = None
        self._buffer_len = buffer_len
        self._quat_buffer = []
        self._acc_buffer = []
        self._is_reading = False
        self._read_thread = None

    def _read(self):
        # num_float_one_frame = 36
        num_float_one_frame = 42
        data_temp = ''
        while self._is_reading:
            data_temp = self._imu_socket.recv(1024).decode('ascii', 'ignore')

            # 正则表达式取出数据
            regex = re.compile(r"\-?\d+\.+\d+")

            data = re.findall(regex, data_temp)
            # 一个IMU有6个数据,四元数通过计算得出
            # 使用7个imu，第七个放在player2的head
            # data = data[0:6*6]
            data = data[0:7 * 6]

            if len(data) >= num_float_one_frame:
                d = np.array(data).reshape((6, 6)).T.astype(float)
                # d = np.array(data).reshape((6, 7)).T.astype(float)  # (7,6)

                quatArray = []
                for l in d[:, 3:6]:
                    x = pi * l[0] / 180 / 2
                    y = pi * l[1] / 180 / 2
                    z = pi * l[2] / 180 / 2
                    q0 = cos(y) * sin(x) * cos(z) + sin(y) * cos(x) * sin(z)
                    q1 = sin(y) * cos(x) * cos(z) - cos(y) * sin(x) * sin(z)
                    q2 = -sin(y) * sin(x) * cos(z) + cos(y) * cos(x) * sin(z)
                    q3 = cos(y) * cos(x) * cos(z) + sin(y) * sin(x) * sin(z)
                    quatArray += [q0, q1, q2, q3]

                # q_share_head = (q_palyer1_head + q_player2_head) / 2
                quatArray[4] = (quatArray[4] + quatArray[6]) / 2
                quatArray = [np.array(quatArray[:6]).reshape((6, 4)).astype(float)]
                # acc_share_head = (acc_player1_head + acc_player2_head) / 2
                acc_share_head=(d[:1,:3]+d[1:2,:3])/2
                acc=np.concatenate(acc_share_head,d[2:6,:3],axis=0)

                tranc = int(len(self._quat_buffer) == self._buffer_len)
                self._quat_buffer = self._quat_buffer[tranc:] + quatArray
                # self._acc_buffer = self._acc_buffer[tranc:] + [-d[:6, :3].astype(float) * 9.8]
                self._acc_buffer = self._acc_buffer[tranc:] + [-acc.astype(float) * 9.8]
                data_temp = data[-1]
                self.clock.tick()

    def start_reading(self):

        if self._read_thread is None:
            self._is_reading = True
            self._read_thread = threading.Thread(target=self._read)
            self._read_thread.setDaemon(True)
            self._quat_buffer = []
            self._acc_buffer = []
            self._imu_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._imu_socket.connect((self.imu_host, self.imu_port))
            self._read_thread.start()
        else:
            print('Failed to start reading thread: reading is already start.')

    def stop_reading(self):
        """
        Stop reading imu measurements.
        """
        if self._read_thread is not None:
            self._is_reading = False
            self._read_thread.join()
            self._read_thread = None
            self._imu_socket.close()

    def get_current_buffer(self):
        """
        Get a view of current buffer.

        :return: Quaternion and acceleration torch.Tensor in shape [buffer_len, 6, 4] and [buffer_len, 6, 3].
        """
        q = torch.tensor(self._quat_buffer, dtype=torch.float)
        a = torch.tensor(self._acc_buffer, dtype=torch.float)
        return q, a

    def get_mean_measurement_of_n_second(self, num_seconds=3, buffer_len=120):
        """
        Start reading for `num_seconds` seconds and then close the connection. The average of the last
        `buffer_len` frames of the measured quaternions and accelerations are returned.
        Note that this function is blocking.

        :param num_seconds: How many seconds to read.
        :param buffer_len: Buffer length. Must be smaller than 60 * `num_seconds`.
        :return: The mean quaternion and acceleration torch.Tensor in shape [6, 4] and [6, 3] respectively.
        """
        save_buffer_len = self._buffer_len
        self._buffer_len = buffer_len
        self.start_reading()
        time.sleep(num_seconds)
        self.stop_reading()
        q, a = self.get_current_buffer()
        # print("q", q)
        self._buffer_len = save_buffer_len
        return q.mean(dim=0), a.mean(dim=0)


def get_input():
    global running, start_recording
    while running:
        c = input()
        if c == 'q':
            running = False
        elif c == 'r':
            start_recording = True
        elif c == 's':
            start_recording = False


if __name__ == '__main__':

    imu_set = IMUSet(buffer_len=40)

    input('Put imu 1 aligned with your body reference frame (x = Left, y = Up, z = Forward) and then press any key.')
    oris = imu_set.get_mean_measurement_of_n_second(num_seconds=3, buffer_len=200)[0][0]  # 初始四元数
    smpl2imu = quaternion_to_rotation_matrix(oris).view(3, 3).t()  # 初始旋转矩阵
    #
    input('\tFinish.\nWear all imus correctly and press any key.')
    for i in range(3, 0, -1):
        print('\rStand straight in T-pose and be ready. The celebration will begin after %d seconds.' % i, end='')
        time.sleep(1)
    print('\rStand straight in T-pose. Keep the pose for 3 seconds ...', end='')

    # 两人动作拼接
    oris, accs = imu_set.get_mean_measurement_of_n_second(num_seconds=3, buffer_len=200)
    oris = quaternion_to_rotation_matrix(oris)  # T-pose下的旋转矩阵
    device2bone = smpl2imu.matmul(oris).transpose(1, 2).matmul(torch.eye(3))  # 初始旋转矩阵乘以T-pose下的旋转矩阵，转置后乘以单位矩阵，取对角线
    acc_offsets = smpl2imu.matmul(accs.unsqueeze(-1))  # [num_imus, 3, 1], already in global inertial frame

    print('\tFinish.\nStart estimating poses. Press q to quit, r to record motion, s to stop recording.')
    imu_set.start_reading()

    server_for_unity = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_for_unity.bind(('127.0.0.1', 8888))
    server_for_unity.listen(1)
    print('Server start. Waiting for unity3d to connect.')
    conn, addr = server_for_unity.accept()

    running = True
    clock = Clock()
    is_recording = False
    record_buffer = None

    get_input_thread = threading.Thread(target=get_input)
    get_input_thread.setDaemon(True)
    get_input_thread.start()

    while running:
        # calibration
        clock.tick(60)
        ori_raw, acc_raw = imu_set.get_current_buffer()  # [1, 6, 4], get measurements in running fps
        ori_raw = quaternion_to_rotation_matrix(ori_raw).view(1, 6, 3, 3)
        acc_cal = (smpl2imu.matmul(acc_raw.view(-1, 6, 3, 1)) - acc_offsets).view(1, 6, 3)
        ori_cal = smpl2imu.matmul(ori_raw).matmul(device2bone)

        # normalization
        acc = torch.cat((acc_cal[:, :5] - acc_cal[:, 5:], acc_cal[:, 5:]), dim=1).bmm(ori_cal[:, -1]) / config.acc_scale
        ori = torch.cat((ori_cal[:, 5:].transpose(2, 3).matmul(ori_cal[:, :5]), ori_cal[:, 5:]), dim=1)
        data_nn = torch.cat((acc.view(-1, 18), ori.view(-1, 54)), dim=1).to(device)
        pose, tran = inertial_poser.forward_online(data_nn)  # 动作建模
        pose = rotation_matrix_to_axis_angle(pose.view(1, 216)).view(72)  # 旋转矩阵转为旋转角

        # recording
        if not is_recording and start_recording:
            record_buffer = data_nn.view(1, -1)
            is_recording = True
        elif is_recording and start_recording:
            record_buffer = torch.cat([record_buffer, data_nn.view(1, -1)], dim=0)
        elif is_recording and not start_recording:
            torch.save(record_buffer, 'data/imu_recordings/r' + datetime.now().strftime('%T').replace(':', '-') + '.pt')
            is_recording = False

        # send pose
        s = ','.join(['%g' % v for v in pose]) + '#' + \
            ','.join(['%g' % v for v in tran]) + '$'
        conn.send(s.encode('utf8'))  # I use unity3d to read pose and translation for visualization here

        print('\r', '(recording)' if is_recording else '', 'Sensor FPS:', imu_set.clock.get_fps(),
              '\tOutput FPS:', clock.get_fps(), end='')

    get_input_thread.join()
    imu_set.stop_reading()
    print('Finish.')
