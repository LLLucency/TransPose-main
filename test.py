import socket
import threading
from articulate.math import *
from datetime import datetime
import torch
import numpy as np
import config
import time
from net import TransPoseNet
from pygame.time import Clock
import re
import math
from math import cos as c
from math import sin as s

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
inertial_poser = TransPoseNet(num_past_frame=60, num_future_frame=5).to(device)
running = False
start_recording = False


class IMUSet:
    r"""
    Sensor order: left forearm, right forearm, left lower leg, right lower leg, head, pelvis
    port原始7002
    """

    # def __init__(self, imu_host='10.208.69.104', imu_port=9090, buffer_len=40):
    def __init__(self, imu_host='10.203.161.205', imu_port=9090, buffer_len=40):

        """
        Init an IMUSet for Noitom Perception Legacy IMUs. Please follow the instructions below.

        Instructions:
        --------
        1. Start `Axis Legacy` (Noitom software).
        2. Click `File` -> `Settings` -> `Broadcasting`, check `TCP` and `Calculation`. Set `Port` to 7002.
        3. Click `File` -> `Settings` -> `Output Format`, change `Calculation Data` to
           `Block type = String, Quaternion = Global, Acceleration = Sensor local`
        4. Place 1 - 6 IMU on left lower arm, right lower arm, left lower leg, right lower leg, head, root.
        5. Connect 1 - 6 IMU to `Axis Legacy` and continue.

        :param imu_host: The host that `Axis Legacy` runs on.
        :param imu_port: The port that `Axis Legacy` runs on.
        :param buffer_len: Max number of frames in the readonly buffer.
        """
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
        """
        The thread that reads imu measurements into the buffer. It is a producer for the buffer.
        """
        num_float_one_frame = 36
        # num_float_one_frame = 30
        data_temp = ''
        while self._is_reading:
            # data_temp += self._imu_socket.recv(1024).decode('ascii', 'ignore')
            data_temp = self._imu_socket.recv(1024).decode('ascii', 'ignore')
            # print(data_temp)
            # print("\n")
            # d1 = np.array(data[2:].split(' ', 41)).reshape((7, 6))
            regex = re.compile(r"\-?\d+\.+\d+")
            data = re.findall(regex, data_temp)
            data = data[0:36]
            # print(len(strs))
            # print(data)
            # print("\n")
            # print(data[2:])

            # print(data)
            if len(data) >= num_float_one_frame:
                # print(np.array(strs[:-3]).reshape((21, 16)))  # full data
                # print('!')
                # d = np.array(data).reshape(6, 7).T  # first 6 imus
                d = np.array(data).reshape((6, 6)).T.astype(float)
                # print(d)

                d = d.tolist()
                for i in range(6):
                    xg = d[i][0]
                    yg = d[i][1]
                    zg = d[i][2]
                    x = d[i][3]
                    y = d[i][4]
                    z = d[i][5]
                    x = math.pi * x / 180 / 2
                    y = math.pi * y / 180 / 2
                    z = math.pi * z / 180 / 2

                    xx = x
                    yy = y
                    zz = z

                    # 角度全取负是为什么
                    x = -xx
                    y = -zz  # 为什么是这样
                    z = -yy

                    # 加速度这样取是为什么
                    d[i][0] = -xg
                    d[i][1] = zg
                    d[i][2] = -yg

                    q0 = math.cos(y) * math.sin(x) * math.cos(z) + math.sin(y) * math.cos(x) * math.sin(z)
                    q1 = math.sin(y) * math.cos(x) * math.cos(z) - math.cos(y) * math.sin(x) * math.sin(z)
                    q2 = -math.sin(y) * math.sin(x) * math.cos(z) + math.cos(y) * math.cos(x) * math.sin(z)
                    q3 = math.cos(y) * math.cos(x) * math.cos(z) + math.sin(y) * math.sin(x) * math.sin(z)
                    '''
                    qq0 = q0
                    qq1 = q1
                    qq2 = q2
                    qq3 = q3
                    q0 = qq3
                    q1 = qq0
                    q2 = qq1
                    q3 = qq2
                    '''
                    d[i][3] = q0
                    d[i][4] = q1
                    d[i][5] = q2
                    d[i].append(q3)
                # d.append([0, 0, 1, 1, 0, 0, 0])
                d = np.array(d)
                tranc = int(len(self._quat_buffer) == self._buffer_len)
                self._quat_buffer = self._quat_buffer[tranc:] + [d[:, 3:7].astype(float)]
                # print(self._quat_buffer)
                self._acc_buffer = self._acc_buffer[tranc:] + [-d[:, 0:3].astype(float) * 9.8]
                data_temp = data[-1]
                self.clock.tick()

    def start_reading(self):
        """
        Start reading imu measurements into the buffer.
        """
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
        # print(q)
        # print("\n")
        # print(a)
        # print(q)
        # print(a)
        # print(a)
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
        # print(q)
        # print("\n")
        # print(a)
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
    imu_set = IMUSet(buffer_len=1)
    #
    input('Put imu 1 aligned with your body reference frame (x = Left, y = Up, z = Forward) and then press any key.')
    print('Keep for 3 seconds ...', end='')
    oris = imu_set.get_mean_measurement_of_n_second(num_seconds=3, buffer_len=200)[0][0]
    # oris = imu_set.get_mean_measurement_of_n_second(num_seconds=3, buffer_len=200)[0].data
    smpl2imu = quaternion_to_rotation_matrix(oris).view(3, 3).t()

    input('\tFinish.\nWear all imus correctly and press any key.')
    for i in range(3, 0, -1):
        print('\rStand straight in T-pose and be ready. The celebration will begin after %d seconds.' % i, end='')
        time.sleep(1)
    print('\rStand straight in T-pose. Keep the pose for 3 seconds ...', end='')
    oris, accs = imu_set.get_mean_measurement_of_n_second(num_seconds=3, buffer_len=200)
    oris = quaternion_to_rotation_matrix(oris)
    device2bone = smpl2imu.matmul(oris).transpose(1, 2).matmul(torch.eye(3))
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
        acc_cal = (smpl2imu.matmul(acc_raw.view(-1, 6, 3, 1)) - acc_offsets).view(1, 6, 3) + torch.tensor([[0, 0, 0],
                                                                                                           [0, 0, 0],
                                                                                                           [0, 0, 0],
                                                                                                           [0, 0, 0],
                                                                                                           [0, 0, 0],
                                                                                                           [0, 0,
                                                                                                            0]]).view(1,
                                                                                                                      6,
                                                                                                                      3)
        ori_cal = smpl2imu.matmul(ori_raw).matmul(device2bone)

        # normalization
        acc = torch.cat((acc_cal[:, :5] - acc_cal[:, 5:], acc_cal[:, 5:]), dim=1).bmm(ori_cal[:, -1]) / config.acc_scale
        ori = torch.cat((ori_cal[:, 5:].transpose(2, 3).matmul(ori_cal[:, :5]), ori_cal[:, 5:]), dim=1)
        data_nn = torch.cat((acc.view(-1, 18), ori.view(-1, 54)), dim=1).to(device)
        pose, tran = inertial_poser.forward_online(data_nn)
        pose = rotation_matrix_to_axis_angle(pose.view(1, 216)).view(72)

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
