import cv2
import time
import numpy as np
from FrameWorker import takeSudoku, takeDigitsAndPutThemInMatrix, sudokuSolver
import Utils
import time
video = cv2.VideoCapture(0)

a = 0
target = 0
sudokuGrid = np.zeros((9, 9))
sudokuGridString = ""
sudokuGridCopy = np.zeros((9, 9))
f = open("Time.txt", "a")
while True:
	a = a + 1
	sudokuGridString = ""
	start = time.time()
	check, frame = video.read()
	TpreluareFrame = time.time()
	frame = cv2.resize(frame, (800, 600))
	cv2.imshow("Originala", frame)
	#print(check)
	#print(frame)
	startSud = time.time()
	img = takeSudoku(frame)
	TstartSud = time.time()

	if img is not None:
		#cv2.imshow("Jocul", img)
		sudokuGridCopy = sudokuGrid
		startMatrix = time.time()
		sudokuGrid = takeDigitsAndPutThemInMatrix(img, frame)
		TstartMatrix = time.time()
		#if np.array_equal(sudokuGrid, sudokuGridCopy) == False:
		#cv2.imshow("Perspectiva Top-Down", img)
		#if sudokuGrid is not None:
			#for i in range(9):
				#for j in range(9):
					#a = str(sudokuGrid[i][j])
					#sudokuGridString += a

			#print(sudokuGridString)
			#sudokuGrid = sudokuSolver(sudokuGridString)
		f.write(str((TpreluareFrame - start) * 1000))
		f.write(", ")
		f.write(str((TstartSud - startSud) * 1000))
		f.write(", ")
		f.write(str((TstartMatrix - startMatrix) * 1000))
		f.write("\n")
			#utils.printGrid(sudokuGrid)
	#else:
		#cv2.imshow("Frame", frame)
	key = cv2.waitKey(1)

	if key == ord('q'):
		break

	#image = frame.copy()
	#time.sleep(0.1)
	
#print(a)
f.close()
video.release()

cv2.destroyAllWindows