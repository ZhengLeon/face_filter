import os
import shutil
import dlib
import cv2

def CreateDir(path):
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path) 
        #print(path+' 目录创建成功')
    #else:
        #print(path+' 目录已存在')


def FilterFace(filepath, newPath):
    fileNames = os.listdir(filepath) 
    for file in fileNames:
        #path = "./picture.txt"
        #f = open(path)
        #lines = f.readlines()
        #print(line)
        #linesplit = line.split('\n')[0]
        newDir = filepath + '/' + file
        if os.path.isfile(newDir):  
            print(newDir)
            newFile = newPath + file

            detector = dlib.get_frontal_face_detector()
            image = cv2.imread(newDir)
            b, g, r = cv2.split(image)
            image_rgb = cv2.merge([r, g, b])
            rects = detector(image_rgb, 1)
            if len(rects) >= 1:
                shutil.copyfile(newDir, newFile)          
        else:
            FilterFace(newDir,newPath)          

if __name__ == "__main__":
    #path = input("输入需要复制文件目录：")
    path = 'E:\\datasets\\CNBC\\Multiracial'
    # 创建目标文件夹
    mkPath = "./all/"
    #CreateDir(mkPath)
    FilterFace(path,mkPath)