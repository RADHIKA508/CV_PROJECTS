import cv2 
import time
from cv2 import CV_32F
import numpy as np
import pandas as pd
import csv


meter_per_frame=1

def write_csv(data):
    with open("displacement_camera.csv", 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(data)

def write_into_csv(data):
    with open("displacement_camera.csv", 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(data)

#sift
orb = cv2.ORB_create()
sift = cv2.SIFT_create()

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)

cap = cv2.VideoCapture(0)
#cap2 = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(0)
img=cv2.imread('photo.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

keypoints_image, descriptors_image = orb.detectAndCompute(img,None)
descriptors_image = descriptors_image.astype(np.float32)

frame_no=0
vel_x, vel_y, vel_z = [], [], []
flag = 0
while cap.isOpened():
    # read images

    suc1, img1 = cap.read()
    fuc1, umg1 = cap.read()
    if suc1:

        if cap.get(cv2.CAP_PROP_POS_MSEC)!=0:
            flag=1

        # if flag==1 and cap.get(cv2.CAP_PROP_POS_MSEC)==0:
        #     break
        time1=float(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
        print("for frame: ",frame_no," Timestamp: ",str(time1))
        frame_no=frame_no+1

        time2=float(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
        suc2, img2 = cap.read()
        fuc2, umg2 = cap.read()
        print("for frame: ",frame_no," Timestamp: ",str(time2))
        frame_no=frame_no+1

        start = time.time()

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        if fuc2 and fuc1:
            umg1 = cv2.cvtColor(umg1, cv2.COLOR_BGR2GRAY)
            umg2 = cv2.cvtColor(umg2, cv2.COLOR_BGR2GRAY)

        keypoints_1, descriptors_1 = orb.detectAndCompute(img1,None)
        keypoints_2, descriptors_2 = orb.detectAndCompute(img2,None)

        descriptors_1 = descriptors_1.astype(np.float32)
        descriptors_2 = descriptors_2.astype(np.float32)
        
        matches = flann.knnMatch(descriptors_1,descriptors_2, k=2)
        matches_image = flann.knnMatch(descriptors_1, descriptors_image, k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]
        matchesMask2 = [[0,0] for i in range(len(matches_image))]

        # ratio test as per Lowe's paper

        temp_x, temp_y, temp_z = [], [], []
        coords1, coords_final = [], []

        for i,mat in enumerate(matches_image):
            if mat[0].distance<200 and mat[0].distance < 0.7*mat[1].distance:
                coords1.append(keypoints_1[mat[0].queryIdx].pt)
                matchesMask2[i]=[1,0]

        for i,mat in enumerate(matches):
            if mat[0].distance < 75 and mat[0].distance < 0.7*mat[1].distance:
                if keypoints_1[mat[0].queryIdx].pt in coords1:
                    matchesMask[i]=[1,0]
                    if fuc2:

                        
                        temp_x.append(
                            keypoints_2[mat[0].trainIdx].pt[0]
                            -
                            keypoints_1[mat[0].queryIdx].pt[0]
                        )

                        temp_y.append(
                            keypoints_2[mat[0].trainIdx].pt[1]
                            -
                            keypoints_1[mat[0].queryIdx].pt[1]
                        )
                        if umg2[int(keypoints_2[mat[0].trainIdx].pt[1])][int(keypoints_2[mat[0].trainIdx].pt[0])]!=0 and umg1[int(keypoints_1[mat[0].queryIdx].pt[1])][int(keypoints_1[mat[0].queryIdx].pt[0])]!=0:
                            z = (
                                umg2[int(keypoints_2[mat[0].trainIdx].pt[1])][int(keypoints_2[mat[0].trainIdx].pt[0])]
                                -
                                umg1[int(keypoints_1[mat[0].queryIdx].pt[1])][int(keypoints_1[mat[0].queryIdx].pt[0])]
                            )

                            temp_z.append(z)
    
        if flag==1 and len(temp_x)>0:
            vel_x.append(np.mean(temp_x)*meter_per_frame)
            vel_y.append(np.mean(temp_y)*meter_per_frame)
            vel_z.append(np.mean(temp_z)*meter_per_frame)
        
                
        draw_params = dict(matchColor = (255,255,0),
                        singlePointColor = (132,0,255),
                        matchesMask = matchesMask,
                        flags = cv2.DrawMatchesFlags_DEFAULT)
        img3 = cv2.drawMatchesKnn(img1, keypoints_1, img2, keypoints_2, matches, None, **draw_params)

        draw_params = dict(matchColor = (255,0,0),
                        singlePointColor = (132,157,0),
                        matchesMask = matchesMask2,
                        flags = cv2.DrawMatchesFlags_DEFAULT)

        img4 = cv2.drawMatchesKnn(img1, keypoints_1, img, keypoints_image, matches_image, None, **draw_params)

        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime

        cv2.putText(img3, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        cv2.imshow('ORB', img3)
        cv2.imshow('ORB IMAGE',img4)

        if cv2.waitKey(5) & 0xFF == 27:
            break
print("The video has ended")

vel_x = np.array(vel_x)
vel_y = np.array(vel_y)
vel_z = np.array(vel_z)

write_into_csv(["Index","Dis_x","Dis_y","Dis_z"])

for i in range(min(len(vel_x),len(vel_y))):
    data=[]
    data.append(i+1)
    data.append(vel_x[i])
    data.append(vel_y[i])
    data.append(vel_z[i])

    write_csv(data)

cap.release()