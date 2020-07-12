import cv2
import numpy as np
import Utils
import math
from tensorflow.keras.models import load_model
import os
import pickle 
import BacktrackingSudoku
import time

#lipsa placa gpu nvidia
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
pickle_in = open("modelDigitRecognition.p","rb")
model = pickle.load(pickle_in)
#path = "C:/Users/dumit/Desktop/Licenta/Sudoku Solver/ImaginiPreluateDinSudoku"
gP = None
warpWithoutFilters = None
pts = None
inputPts = None



def takeSudoku(img):
	global warpWithoutFilters
	global gP
	global pts
	global inputPts

	img = cv2.resize(img, (800,600))

	#original = img.copy()

	#cv2.imshow('original', img)

	img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

	greymain = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

	#blur = cv2.GaussianBlur(greymain, (5, 5), 0)

	proc = cv2.GaussianBlur(greymain, (9, 9), 0)
	proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 39, 10)

	proc = cv2.bitwise_not(proc, proc) 
	kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
	proc = cv2.dilate(proc, kernel)

	th2 = cv2.adaptiveThreshold(greymain,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
	            cv2.THRESH_BINARY_INV,39,10)

	#cv2.imshow("th2", th2)

	contours = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

	max_area = 0
	c = 0
	biggest = np.array([], dtype = "float32")
	for cnt in contours:
	        area = cv2.contourArea(cnt)
	        if area > 5000:
	            peri = cv2.arcLength(cnt, True)
	            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
	            if area > max_area and len(approx) == 4:
	                max_area = area
	                best_cnt = cnt
	                img = cv2.drawContours(img, contours, c, (0, 255, 0), 3)
	                biggest = approx
	                
	        c+=1
	#cv2.imshow("BiggestContour", img)

	#print(biggest)
	if len(biggest) >= 4:

		#sortez toate punctele din cadran ca sa fie clockwise
		#print(biggest[0][0][1]
		for i in range (0, len(biggest)):
			for j in range (0, len(biggest)):
				if biggest[i][0][1] < biggest[j][0][1]:
					copy = biggest[i][0].copy()
					biggest[i][0] = biggest[j][0]
					biggest[j][0] = copy
		if biggest[0][0][0] > biggest[1][0][0]:
		    copy = biggest[0][0].copy()
		    biggest[0][0] = biggest[1][0]
		    biggest[1][0] = copy
		if biggest[2][0][0] > biggest[3][0][0]:
		    copy = biggest[2][0].copy()
		    biggest[2][0] = biggest[3][0]
		    biggest[3][0] = copy


		x, y, w, h = cv2.boundingRect(best_cnt)

		pts2 = []

		for i in range(0, len(best_cnt)):
			for j in range(0, len(best_cnt[i])):
				pts2.append(best_cnt[i][j])
				#print(best_cnt[i][j], " ")
				

		# compute the width of the new image, which will be the
		# maximum distance between bottom-right and bottom-left
		# x-coordiates or the top-right and top-left x-coordinates
		widthA = np.sqrt(((biggest[3][0][0] - biggest[2][0][0]) ** 2) + ((biggest[3][0][1] - biggest[2][0][1]) ** 2))
		widthB = np.sqrt(((biggest[1][0][0] - biggest[0][0][0]) ** 2) + ((biggest[1][0][1] - biggest[0][0][1]) ** 2))
		maxWidth = max(int(widthA), int(widthB))
		# compute the height of the new image, which will be the
		# maximum distance between the top-right and bottom-right
		# y-coordinates or the top-left and bottom-left y-coordinates
		heightA = np.sqrt(((biggest[1][0][0] - biggest[3][0][0]) ** 2) + ((biggest[1][0][1] - biggest[3][0][1]) ** 2))
		heightB = np.sqrt(((biggest[0][0][0] - biggest[2][0][0]) ** 2) + ((biggest[0][0][1] - biggest[2][0][1]) ** 2))
		maxHeight = max(int(heightA), int(heightB))

		#print(maxWidth)
		#print(maxHeight)

		pts = np.float32([[0, 0], [maxWidth - 1, 0], [0, maxHeight - 1], [maxWidth - 1, maxHeight - 1]])

		#punctele unde se detecteaza sudokul
		#inputPts = np.float32([[biggest[0][0][0], biggest[0][0][1]], [biggest[1][0][0], biggest[1][0][1]], [biggest[2][0][0], biggest[2][0][1]], [biggest[3][0][0], biggest[3][0][1]]])
		inputPts = np.float32(biggest)


		gP = cv2.getPerspectiveTransform(inputPts, pts)
		
		wP = cv2.warpPerspective(th2, gP, (maxWidth, maxHeight))
		warpWithoutFilters = cv2.warpPerspective(img, gP, (maxWidth, maxHeight))

		#cv2.imshow("warpPerspective", wP)

		return wP




def takeDigitsAndPutThemInMatrix(img, frame):
	global warpWithoutFilters
	global gP
	global pts
	global inputPts

	f = open("Time2.txt", "a")

	wpcopy = img
	imgH, imgW = img.shape
	start = time.time()
	wP = cv2.resize(img, (288, 288))
	#cv2.imshow("beforebit", wP)
	wP = cv2.bitwise_not(wP)
	#cv2.imshow("warpPerspectiveSmall", wP)


	#variabile pentru segmentare
	imaginiPatrate = []
	lat = 32
	inalt = 32

	height = wP.shape[0] // 9
	width = wP.shape[1] // 9

	offsetW = math.floor(width / 10)
	offsetH = math.floor(height / 10)

	#grila pentru Sudoku
	#sudokuGrid = [[0 for x in range(9)] for y in range(9)]
	sudokuGrid = np.zeros((9, 9))
	end = time.time()
	f.write(str((end - start) * 1000))
	f.write(", ")
	#
	#segmentez imaginea in 81 de patrate pentru a putea detecta digiti
	start = time.time()
	for i in range(9):
		for j in range(9):
			#cropImage = wP[inalt * i: inalt * (i + 1), lat * j: lat * (j + 1)]
			#cropImage = wP[lat * i: lat * (i + 1), inalt * j: inalt * (j + 1)]
			cropImage = wP[inalt * i + offsetH: inalt * (i + 1) - offsetH, lat * j + offsetW: lat * (j + 1) - offsetW]
			#print(lat * i, lat * (i + 1), inalt * j, inalt * (j + 1))
			#fileName = str(i) + str(j)
			#print(fileName)
			#finalPath = path + '/' + fileName + ".jpg"
			#cv2.imwrite(path + "/0-0.jpg", wP[0:32, 0:32])
			#cv2.imshow("0-32", wP[0:32, 0:32])
			#cv2.imshow("0-32", wP[0:32, 32:64])
			
			#cv2.imwrite(os.path.join(path , fileName), cropImage)

			#iau cea mai mare componenta conectata
			#cropImage = cv2.bitwise_not(cropImage)
			#cropImage = Utils.largest_connected_component(cropImage)

			#redimensionare
			cropImage = cv2.resize(cropImage, (32, 32))
			
			#cv2.imwrite(finalPath, cropImage)
			#cv2.imwrite(path + "/wp.jpg", wP)	
			totalNumberOfPixels = 32 * 32
			zeroPixels = totalNumberOfPixels - cv2.countNonZero(cropImage)
			#daca sunt mai multi de 9 pixeli negri, continuam sa vedem ce numar avem, daca nu sunt inseamna ca e casuta goala,avand doar zgomot
			#if zeroPixels > 9:
			#print(zeroPixels, i, j)
			cropImage = Utils.prepare(cropImage)

			classId = int(model.predict_classes(cropImage))

			#model = load_model("mnist_keras_cnn_model.h5", compile = True)
			prediction = model.predict([cropImage])
			probVal= np.amax(prediction)
			if probVal < 0.9:
				sudokuGrid[i][j] = 0
			else:
				#print(fileName, classId,probVal)
				#print(i, j, classes)
				sudokuGrid[i][j] = classId
	#Utils.printGrid(sudokuGrid)
	end = time.time()
	f.write(str((end - start) * 1000))
	f.write(", ")
	start = time.time()
	sudokuGridCopy = sudokuGrid.copy()
	warpZero = np.zeros(warpWithoutFilters.shape, np.uint8)
	if valid_board(sudokuGrid) == True:
		wP = cv2.bitwise_not(wP) 
		solutionGrid = sudokuSolver(sudokuGrid)
		#print("Solution:")
		Utils.printGrid(sudokuGrid)
		for i in range(9):
			for j in range(9):
				if sudokuGridCopy[i][j] != sudokuGrid[i][j]:
					y = inalt * i + 22
					x = lat * j + 14
					xWarpZero = int(warpZero.shape[1] / 9 * j + 14)
					yWarpZero = int(warpZero.shape[0] / 9 * i + 22)
					color = (255, 255, 0) 
					cv2.putText(warpZero, str(int(sudokuGrid[i][j])), (xWarpZero, yWarpZero), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
					cv2.putText(wP, str(int(sudokuGrid[i][j])), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
					#print(str(sudokuGrid[i][j]))
		#cv2.imshow("SudokuSolved", wP)
		#cv2.imshow("SolvedSudoku", warpZero)
		#cv2.imwrite("SolvedSudoku.png", warpZero)
		wP = cv2.resize(wP, (imgW, imgH))
		#frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
		#wP = cv2.warpPerspective(frame, gP, (800, 600), cv2.WARP_INVERSE_MAP)
		#frame = cv2.bitwise_not(frame)
		if warpWithoutFilters is not None:
			#print(warpWithoutFilters.shape)
			#print(frame.shape)
			#frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
			#frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
			#warpWithoutFilters = cv2.cvtColor(warpWithoutFilters, cv2.COLOR_BGR2RGB)
			#warpWithoutFilters = cv2.cvtColor(warpWithoutFilters,cv2.COLOR_RGB2GRAY)
			#warpWithoutFilters = cv2.warpPerspective(frame, gP, (800, 600), cv2.WARP_INVERSE_MAP)

			gP = cv2.getPerspectiveTransform(pts, inputPts)
			wPBack = cv2.warpPerspective(warpZero, gP, (800, 600))
			wPBack = cv2.bitwise_not(wPBack)
			wPBack = cv2.bitwise_and(wPBack,frame)
			#frame = cv2.bitwise_and(frame, warpWithoutFilters)
			cv2.imshow("Solution", wPBack)
	else:			
		print("No solution")
	end = time.time()
	f.write(str((end - start) * 1000))
	f.write("\n")
	f.close()
	print("\n")



#functie care pune o masca peste jocul sudoku din imagine
def showMaskOfSudoku(img, best_cnt):
	greymain = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	th2 = cv2.adaptiveThreshold(greymain,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY_INV,39,10)

	mask = np.zeros((greymain.shape),np.uint8)

	cv2.drawContours(mask,[best_cnt],0,255,-1)
	cv2.drawContours(mask,[best_cnt],0,0,2)
	cv2.imshow("mask", mask)


#functie care arata toate contururile cu aria mai mare de 5000
def showAllCountours(img):
	greymain = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	th2 = cv2.adaptiveThreshold(greymain,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY_INV,39,10)

	mask = np.zeros((greymain.shape),np.uint8)

	out = np.zeros_like(th2)
	out[mask == 255] = th2[mask == 255]
	#cv2.imshow("maskOut", out)

	blur = cv2.GaussianBlur(out, (5,5), 0)
	#cv2.imshow("blur1", blur)

	th2 = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
	#cv2.imshow("thresh1", th2)

	contours= cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

	#se trag toate contururile a caror suprafata este mai mare decat 5000 de pixeli
	c = 0
	for i in contours:
	        area = cv2.contourArea(i)
	        if area > 5000:
	            cv2.drawContours(img, contours, c, (0, 255, 0), 3)
	        c+=1
	cv2.imshow("ContourImage", img)

def sudokuSolver(sudokuGridS):

	grid = BacktrackingSudoku.solve(sudokuGridS)
	#utils.printGrid(sudokuGrid)
	return grid



#Functie care veirifca daca coloana este valida
# -1 valori invalide
# 0 valori repetate
# 1 coloana valida
def valid_col(col, grid):
  # Iau coloana
  temp = [row[col] for row in grid]
  #elimin 0
  temp = list(filter(lambda a: a != 0, temp))
  # Verific pentru invalide
  if any(i < 0 and i > 9 for i in temp):
    print("Invalid value")
    return -1
  # Verific pentru cifre egale
  elif len(temp) != len(set(temp)):
    return 0
  else:
    return 1

#pentru linie
def valid_row(row, grid):
  temp = grid[row]
  temp = list(filter(lambda a: a != 0, temp))
  if any(i < 0 and i > 9 for i in temp):
    print("Invalid value")
    return -1
  elif len(temp) != len(set(temp)):
    return 0
  else:
    return 1



# Functie pentru verificarea valididatii a subpatratelor
# -1 pentru valori nevalide
# 0 valori egale
# 1 subpatrat valid
def valid_subsquares(grid):
  for row in range(0, 9, 3):
      for col in range(0,9,3):
         temp = []
         for r in range(row,row+3):
            for c in range(col, col+3):
              if grid[r][c] != 0:
                temp.append(grid[r][c])
          # Valori nevalide
         if any(i < 0 and i > 9 for i in temp):
             print("Invalid value")
             return -1
         # Valori care se repeta
         elif len(temp) != len(set(temp)):
             return 0
  return 1
# Functie pentru verificarea validitatii matricei 
def valid_board(grid):
  # Verifica fiecare linie si coloana
  for i in range(9):
      res1 = valid_row(i, grid)
      res2 = valid_col(i, grid)
      # Daca o linie sau coloana sunt nevalide, toata matricea este nevalida
      if (res1 < 1 or res2 < 1):
          return False
  # Daca liniile si coloane sunt valide, verifica subpatratele
  res3 = valid_subsquares(grid)
  if (res3 < 1):
      return False
  else:
      return True

