import sys
sys.path.append('./')
import serial
import time
import threading
import numpy as np
from numpy.linalg import norm as norm 
from math import inf as inf
from math import pi as pi
from math import pow as pow
from math import tan as tan
from math import sin as sin
from math import cos as cos
from math import atan as atan
from math import atan2 as atan2
from RoboMath import*   #always needed
from Config import*     #always needed

import lidarVector

__all__ = ['Lidar']

class Lidar():
    """
        SINGLE LIDAR OBJECT (non static, for instances use Create function).\n
        Each instance executes in new thread (when Start call)
    """

    _rndKey = 9846    #it is just a random number
    _NumOfLidars = 0

    def __init__(self, rndKey, lidarID):
        "Use create function only!!!"

        if (rndKey != Lidar._rndKey):
            print("Use Lidar.Create function only!!!", exc = True)

        #PORT
        self.ser = serial.Serial(         
            baudrate = 128000,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            exclusive = True,
            timeout = 1.0
            )

        #IDs
        self.lidarID = lidarID
        self.lidarSN = mainDevicesPars[f"lidarSN_ID{self.lidarID}"]
        Lidar._NumOfLidars += 1
        self._frameID = 0   #all captured frames
        self.frameID = 0    #the last public (uploaded) frame

        #MAIN PARAMETERS
        self.mount = np.zeros([2])    #local coordinates
        for i, el in enumerate(mainDevicesPars[f"lidarMount_ID{self.lidarID}"].split()):
            if i > 1:
                break
            self.mount[i] = float(el)
        self.mountPhi = float(mainDevicesPars[f"lidarMountPhi_ID{self.lidarID}"])
        self.range = float(mainDevicesPars[f"lidarRange_ID{self.lidarID}"])
        self.half_dphi = float(mainDevicesPars[f"lidarHalf_dPhi_ID{self.lidarID}"])
        """in degrees"""
        self.phiFrom = 180.0 - (float(mainDevicesPars[f"lidarPhiTo_ID{self.lidarID}"]) - self.mountPhi - self.half_dphi)
        """on this lidar counterclockwise is positive and 180.0 deg. shifted angle"""
        self.phiTo = 180.0 - (float(mainDevicesPars[f"lidarPhiFrom_ID{self.lidarID}"]) - self.mountPhi - self.half_dphi)
        """on this lidar clockwise is positive and 180.0 deg. shifted angle"""

        self.N = round(1.1 * (self.phiTo - self.phiFrom) / (self.half_dphi * 2.0) ) + 40 + 6 # 6 - for edge shifting

        self.phiFromPositive = True
        if (self.phiFrom < 0.0):
            self.phiFrom += 360.0
            self.phiFromPositive = False
        if (self.phiTo < 0.0):
            self.phiTo += 360
        
        #PRIVATE
        self._xy = np.zeros([3, self.N], dtype = np.float64)
        self._xy[2, :] = 1.0
        self._phi = np.zeros([self.N], dtype = np.float64)
        self._Npnts = np.array([self.N], dtype = np.uint32)

        self._linesXY = np.zeros([3, self.N], dtype = np.float64)
        self._linesXY[2, :] = 1.0
        self._Nlines = np.zeros([1], dtype = np.uint32)
        self._gapsIdxs = np.zeros([self.N], dtype = np.uint32)
        self._Ngaps = np.zeros([1], dtype = np.uint32)

        #Cpp extension initialization
        self._cppPars = (   float(globalPars["half_width"]), float(globalPars["half_length"]), float(globalPars["safetyBox"]) / 100.0, \
                            self.range, self.half_dphi, float(mainDevicesPars[f"lidarRegressionTolerance_ID{self.lidarID}"]), \
                            self.mount)
        """(half_width, half_length, safety, range, half_dphi, tolerance, mount"""
        self.cppID = lidarVector.init(self._xy, self._phi, self._Npnts, self._linesXY, self._Nlines, self._gapsIdxs, self._Ngaps, self._cppPars)

        if (self.cppID < 0):
            print("Bad inputs for cpp extension", exc = True)  

        #PUBLIC
        self.edgeShift = lidarVector.getEdgeShift(self.cppID)

        #LOCKS
        self._thread = []
        self._mutex = threading.RLock()

        #EVENTS AND HANDLES
        self.ready = threading.Event()
        self.ready.clear()
        self._isFree = threading.Event()
        self._isFree.set()

    def _Start_wrp(self, f):
        def func(*args, **kwargs):
            if (self._isFree.wait(timeout = 0.5)):  #for the case of delayed stopping
                self._thread = threading.Thread(target = f, args = args, kwargs = kwargs)
                self._thread.start()
                while self._thread.isAlive() and not self.ready.is_set():
                    time.sleep(0.01)
                if self.ready.is_set():
                    time.sleep(0.5) #wait for rotation to become stable
            return self.ready.is_set()
        return func

    def Start(self):
        """
            Starts the lidar forever.\n
            Gets wrapped by self._Start_wrp func which invokes for that a new thread and returns True if succeed, otherwise returns False.\n
            If pinsProcessing = False it blocks pins transform calculations. Memory for pins was allocated while creating, so you can change processing flag further on using Stop/Start\n
            Safe for multiple call.
        """

        self._isFree.clear()

        try:
            port_n = 0
            portDetected = False
            while not portDetected: #finding port with Lidar
                if (port_n > 4):
                    print(f"Cannot find Lidar {self.lidarID} ({self.lidarSN}) on COM", log = True, exc = True)
                try:
                    self.ser.port='/dev/ttyUSB' + str(port_n)
                    if self.ser.is_open:
                        raise Exception(msg = 'Busy port')
                    else:
                        self.ser.open()

                    portDetected = True
                    self.ser.write(b'\xa5\x65')
                    self.ser.flush()
                    self.ser.reset_input_buffer()
                    time.sleep(0.1)
                    self.ser.reset_input_buffer()   #old data confusely remains on the device
                    time.sleep(0.1)
                    self.ser.reset_input_buffer()
                    self.ser.write(b'\xa5\x90')
                    self.ser.flush()
                    t0 = time.time()
                    while(not self.ser.read(11) or self.ser.read(16).hex() != self.lidarSN):
                        if (time.time() - t0 > 0.3):
                            portDetected = False
                            self.ser.close()
                            break
                except:
                    portDetected = False
                finally:
                    port_n += 1
            #Время на раскручивание головки. Сразу после открытия порта (это автоматически запускает съемку и идут полезные пакеты облаков) идет команда остановить скан (головка продолжает вращаться), 
            #потом через время команда запустить скан и пропускаются 19 байт (7 + 12), относящиеся к ответу на запуск. 
            self.ser.write(b'\xa5\x65')
            self.ser.flush()
            time.sleep(0.2)
            self.ser.reset_output_buffer()
            self.ser.reset_input_buffer()
            self.ser.write(b'\xa5\x60')  
            skippedBytes = self.ser.read(19) #Далее идут полезные пакеты с облаками

            def AngCorrect(dist):
                if (dist):
                    return 57.295779513 * atan(0.0218 / dist - 0.14037347)  #in degrees
                else:
                    return 0.0

            angles = np.zeros(self.N // 2)
            dists = np.zeros(self.N // 2)
            
            diffAngle = 0.0
            n = 0

            self.ready.set()
            while (self.ready.is_set() and threading.main_thread().is_alive()):

                t0 = time.time()
                while(self.ser.read() != b'\xaa' or self.ser.read() != b'\x55'):
                    n = 0   #если случился проскок, данные должны записываться сначала - и то не факт!
                    # print('LIDAR DATA SLIP', log = True)
                    if (time.time() - t0 > self.ser.timeout):
                        print(f"Lidar {self.lidarID} does not response", log = True, exc = True)

                mode = self.ser.read()[0]    #можно использовать для определения круга (где-то на 346 градусах почему-то выставляется 1)
                length = self.ser.read()[0]
                angles[0] = (int.from_bytes(self.ser.read(2), 'little') >> 1) / 64.0
                angles[length - 1] = (int.from_bytes(self.ser.read(2), 'little') >> 1) / 64.0
                checkCode = self.ser.read(2)
                buff = self.ser.read(2 * length)
                for i in range(length):
                    dists[i] = int.from_bytes(buff[2 * i : 2 * (i + 1)], 'little') / 4000.0

                if (n + length >= self.N):   #предупрежжение выхода за пределы массива, размеченного под облако
                    n = 0 
                    continue

                if (length > 1):
                    if (angles[length - 1] < angles[0]):
                        diffAngle = (angles[length - 1] + 360.0 - angles[0]) / (length - 1)
                    else:
                        diffAngle = (angles[length - 1] - angles[0]) / (length - 1)
                else:
                    diffAngle = 0.0

                isRound = False
                angles0 = angles[0]
                for i in range(length):
                    angles[i] = diffAngle * i + angles0 + AngCorrect(dists[i])

                    if (angles[i] > 360.0):
                        angles[i] -= 360.0
                        
                        #ROUND
                        if (not isRound and n): #возможен пропуск окончания круга, если проход оборота (360 град.) попадает аккурат между пакетами, но такого пока замечено не было
                            #!!!Нужно помнить, что начало круга из-за коррекции угла может весьма ощутимо плавать - до + нескольких градусов к желаемому PhiFrom

                            if (self._xy[0, 0] == 0.0 and self._xy[1, 0] == 0.0):   #лидар отвратительно измеряет углы, если нет дистанции, а нач. и кон. угол очень важны
                                self._phi[0] = 0.01745329252 * (180.0 - (self.phiFrom - self.mountPhi))
                            if (self._xy[0, n - 1] == 0.0 and self._xy[1, n - 1] == 0.0):
                                self._phi[n - 1] = 0.01745329252 * (180.0 - (self.phiTo - self.mountPhi)) # + angleOffset

                            with self._mutex:
                                self._Npnts[0] = n
                                lidarVector.calcLines(self.cppID)
                                lidarVector.synchronize(self.cppID)
                                self._frameID += 1

                            n = 0
                            isRound = True

                    if (self.phiFromPositive and (angles[i] >= self.phiFrom and angles[i] < self.phiTo)) or \
                        (not self.phiFromPositive and (angles[i] >= self.phiFrom or angles[i] < self.phiTo)):

                        self._phi[n] = 0.01745329252 * (180.0 - (angles[i] - self.mountPhi)) # + angleOffset

                        if (dists[i] > 0.02 and dists[i] < self.range):    #it is the limit of lidar itself
                            self._xy[0, n] = dists[i] * cos(self._phi[n])
                            self._xy[1, n] = dists[i] * sin(self._phi[n])
                        else:
                            self._xy[:2, n] = 0.0
                        n += 1
        
        except:
            if (sys.exc_info()[0] is not RoboException):
                print('Lidar ' + str(self.lidarID) + ' error! ' + str(sys.exc_info()[1]) + '; line: ' + str(sys.exc_info()[2].tb_lineno), log = True)
        finally:
            self.ready.clear()
            if self.ser.is_open:
                # print('Lidar port closing')
                self.ser.close()
            self._isFree.set()
    
    def Stop(self):
        """Stops the lidar"""
    ####WE HAVE TO USE NON-BLOCKING STOP BECAUSE OF SOME WORK LOGIC MOMENTS, TO BE RELEVANT IN START THERE IS A _isFree.wait() ON START
        self.ready.clear()
        # self._isFree.wait(timeout = 5.0) 

    @classmethod
    def Create(cls, lidarID):
        "Start lidar the first one of any other devices!!!"
        try:
            lidar = Lidar(cls._rndKey, lidarID)
            lidar.Start = lidar._Start_wrp(lidar.Start)
            return lidar
        except:
            if (sys.exc_info()[0] is not RoboException):
                print(f"Lidar {lidar.lidarID} error! {sys.exc_info()[1]}; line: {sys.exc_info()[2].tb_lineno}")
            return None

    def Release(self):
        print("py releasing")
        lidarVector.release(self.cppID)

    def GetLinesXY(self, linesXY, pauseIfOld = 0.0):
        with self._mutex:
            if not self.ready.is_set():
                return -1
            if self.frameID != self._frameID:
                self.frameID = self._frameID
                return linesXY.Fill(self._linesXY, self._gapsIdxs, int(self._Nlines[0]), int(self._Ngaps[0]))
        time.sleep(pauseIfOld)
        return 0 #the trick is - even if the pointcloud is empty there are 3 elements in linesXY