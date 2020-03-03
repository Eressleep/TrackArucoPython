import numpy as np
import cv2
import cv2.aruco as aruco
import sys,time,math

#настройка метки
id_to_find = 72
marer_size = 10 #~сm

#калибровка камеры
calib_path = ""
camera_matrix = np.loadtxt(calib_path + 'cameraMatrix.txt',delimiter=',')
camera_matrix = mass_matrix
camera_distortion = np.loadtxt(calib_path + 'cameraDistortion.txt',delimiter=',')

#---
R_flip  = np.zeros((3, 3), dtype=np.float32)
R_flip[0, 0] = 1.0
R_flip[1, 1] = -1.0
R_flip[2, 2] = -1.0

#объявление метки
aruco_dist = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
parameters =  aruco.DetectorParameters_create()

#юзаем камеру
cap = cv2.VideoCapture(0)
#настроем размер камеры
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1500)

#----
font = cv2.FONT_HERSHEY_PLAIN

setpoint = float

def pid():


    return 2


while True:
    #читаем столбцы и строки с камеры
    ret, frame = cap.read()

    #Конвертируем в серые тона
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #поиск всех маркеров
    corners, ids, rejected = aruco.detectMarkers(image=gray,dictionary=aruco_dist, parameters=parameters,
                                                 cameraMatrix=camera_matrix, distCoeff=camera_distortion)
    if ids != None and ids[0] ==id_to_find:
        ret = aruco.estimatePoseSingleMarkers(corners, marer_size, camera_matrix, camera_distortion)

        rvec, tvec = ret[0][0, 0, :], ret[1][0, 0, :]
        #обрисуем маркер
        aruco.drawDetectedMarkers(frame, corners)
        aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec, tvec, 10)
        #---
        str_position = "cord x = %4.0f  y=%4.0f  z=%4.0f" % (tvec[0], tvec[1], 5)
        str_position = "cord x = %4.0f  y=%4.0f  z=%4.0f" % (tvec[0], tvec[1], 5)
        cv2.putText(frame, str_position, (0, 100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        #self.pos_x = x - tvec[0] * const
        #self.pos_y = y - tvec[1] * const
        #
    cv2.imshow('frame', frame)
    #---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break