import cv2
import numpy as np
import asyncio
import face_recognition as fr
import time
import string
import random
from pathlib import Path
import json
import imutils
#from picamera.array import PiRGBArray
#from picamera import PiCamera


#import gpiozero

#relay = gpiozero.LED(14)

def accessGranted(name=None):
	#relay.blink(5, 1, 1)
	print('открыто')
	if name:
		print('Привет',name)


def accessDenied(name=None):
	#relay.off()
	print('Закрыто')
	pass

# ---------------------------------------------------------------------------------------

async def videoProcessing(identifier, imshow=False):

	picam = cv2.VideoCapture(2)  #камера ноута
	picam.set(3,640) # set Width
	picam.set(4,480) # set Height
	#picam = PiCamera()  # камера малинки
	#picam.resolution = ( 1280, 720 )
	#raw = PiRGBArray(picam)

	print('started video stream')
	await asyncio.sleep(0.1)

	while True:
		#Ожидание внешнего сигнала
		await asyncio.sleep(0.1)
		if identifier.exit:
			break
	
		print('распознавание')
		try:
                        _, frame = picam.read()
                        frame = imutils.resize(frame, width=300)
		except Exception as e:
			print(e)
			continue


		scaled = cv2.resize(frame,None, fx=0.5, fy=0.5)
		face_locations = fr.face_locations(scaled)
		

		for top,right,bottom,left in face_locations:
			
			# Выделение лица
			cv2.rectangle(scaled,(left, top), (right, bottom), (255,0,0), 3)

			top *= 2
			bottom *= 2
			right *= 2
			left *= 2

			face_img = frame[top:bottom, left:right] #извлекаем лицо


			try:
				await asyncio.sleep(0.1)	
				face_encoding = fr.face_encodings(face_img)[0]

			except Exception as e:
				# Если не распознается - просто продолжаем
				continue

			person = identifier.getIDFromEncoding(face_encoding)

			if person is None:
				# Новое лицо. Добавляем и сохраняем
				print('Найдено новое лицо')
				identifier.addNew(face_img, face_encoding)
				continue

			if identifier.hasAccess(person):
				accessGranted()
			else:
				accessDenied()

		ret, v = cv2.imencode('.jpg', scaled)
		identifier.setView(v)


	#picam.close()
	picam.release()
	cv2.destroyAllWindows()

