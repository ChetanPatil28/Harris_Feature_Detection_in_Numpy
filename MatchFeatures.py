import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import filters
import matplotlib.pyplot as plt

img1=cv2.imread("transA.jpg")
img1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

img2=cv2.imread("transB.jpg")
img2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

def harris_R(img,sig=2):
    imx=np.zeros(img.shape)
    imy=np.zeros(img.shape)
    filters.gaussian_filter(img,(sig,sig),(0,1),imx)
    filters.gaussian_filter(img,(sig,sig),(1,0),imy)
    Wxx=filters.gaussian_filter(imx*imx,sig)
    Wyy=filters.gaussian_filter(imy*imy,sig)
    Wxy=filters.gaussian_filter(imx*imy,sig)
    Wdet=Wxx*Wyy-Wxy**2
    Wtrace=Wxx+Wyy
    return Wdet/Wtrace

def find_filtered_cords(resp_img,thresh=0.1,min_dist=10):
    crnr_t=resp_img.max()*0.1
    resp_t=(resp_img>crnr_t)*1
    cords=np.array(resp_t.nonzero()).T
    resp_cords=resp_img[cords[:,0],cords[:,1]]
    indices= np.argsort(resp_cords)
    allowed_locations= np.zeros(resp_t.shape)
    allowed_locations[min_dist:-min_dist,min_dist:-min_dist] = 1
    filtered_cords=[]
    for i in indices: 
        if allowed_locations[cords[i,0],cords[i,1]] == 1:
            filtered_cords.append(cords[i]) 
            allowed_locations[(cords[i,0]-min_dist):(cords[i,0]+min_dist), (cords[i,1]-min_dist):(cords[i,1]+min_dist)] = 0

    return filtered_cords

def get_descriptors(image,filtered_coords,wid=5):
    desc = [] 
    for coords in filtered_coords: 
        patch = image[coords[0]-wid:coords[0]+wid+1, coords[1]-wid:coords[1]+wid+1].flatten() 
        desc.append(patch)
    return desc


def match(desc1,desc2,th=0.5):
    n = len(desc1[0])
    z1 = ((desc1-np.mean(desc1,axis=1).reshape(-1,1))/np.std(desc1,axis=1).reshape(-1,1))
    z2 = ((desc2-np.mean(desc2,axis=1).reshape(-1,1))/np.std(desc2,axis=1).reshape(-1,1))
    d=np.dot(z1,z2.T)/n
    d=np.where(d>th,d,-1)
    ndx = np.argsort(-d) 
    matchscores = ndx[:,0]
    return matchscores


def visualizeMatches(img1,img2,matchscores,filt_cords1,filt_cords2):
    full_img=np.hstack((img1,img2))
    full_img=cv2.cvtColor(full_img,cv2.COLOR_GRAY2BGR)
    for i in matchscores:
        col=np.random.randint(0,256,size=3)
        #print(x[0])
        b,a=filtered_cords2[i]
        y,x=img2.shape
        b,a=b,a+x
        cv2.line(full_img,tuple(filtered_cords1[i][::-1]),(b,a)[::-1],(int(col[1]),int(col[1]),int(col[2])),1)
    cv2.imshow('full',full_img)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    resp1=harris_R(img1)
    resp2=harris_R(img2)
    filtered_cords1=find_filtered_cords(resp1)
    filtered_cords2=find_filtered_cords(resp2)
    desc1=get_descriptors(img1,filtered_cords1,wid=5)
    desc2=get_descriptors(img2,filtered_cords2,wid=5)
    matches=match(desc1,desc2,th=0.5)
    visualizeMatches(img1,img2,matches,filtered_cords1,filtered_cords2)