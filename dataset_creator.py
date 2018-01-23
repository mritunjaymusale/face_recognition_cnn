import numpy as np
import cv2 
import os
from scipy.ndimage import imread
import face_recognition
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import train_test_split
datadir = 'dataset/'

imageLabel=[]

#loop for gettin all the data present in the dataset folder
for i in os.listdir(datadir):
	for subdir, dirs, files in os.walk(os.path.join(datadir,i)):
		#perform data extraction  here
		#'files' returns a list of all the files in the directory 
		for file in files:
			
			# image = face_recognition.load_image_file(os.path.join(subdir,file))
			# face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0)
			# print("I found {} face(s) in this photograph.".format(len(face_locations)))
			# if len(face_locations) == 1 :
			# 	for face_location in face_locations:
			# # location of each face in this image
			# 		top, right, bottom, left = face_location

			# # You can access the actual face itself like this:
			# 		face_image = image[top:bottom, left:right]
			# 		cv2.imwrite('temp.png',face_image)
			# 		face_image = cv2.imread("./temp.png")
			# 		face_image = cv2.resize(face_image,(150,150), interpolation = cv2.INTER_AREA)
			# 		face_image = cv2.cvtColor(face_image,cv2.COLOR_BGR2RGB)
			# 		cv2.imwrite(os.path.join(subdir,file),face_image)

			# else:
			
			# 	os.remove(os.path.join(subdir,file))
				
				

	#perform label extraction here
			
			imageLabel.append(int(i))

####one-hot-array cretor (used in classification problem)
labelbinarizer = LabelBinarizer()
#that '+1' at the end might cause problems just take of it the future
labelbinarizer.fit(range(max(imageLabel)+1))
one_hot_encoded_data = labelbinarizer.transform(imageLabel)

x_data =[] 

for i in os.listdir(datadir):
    for subdir, dirs, files in os.walk(os.path.join(datadir,i)):
        for file in files:

            temp_img = imread(os.path.join(subdir,file))
            x_data.append(temp_img)

x_data =np.array(x_data)
y_data = one_hot_encoded_data
 
X_train,X_test,Y_train,Y_test = train_test_split(x_data,y_data,train_size=0.9,random_state=2)
np.savez('data.npz',X_train=X_train,X_test=X_test,Y_train=Y_train,Y_test=Y_test)