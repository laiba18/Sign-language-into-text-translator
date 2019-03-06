import numpy as np
import cv2
import glob
from numpy.linalg import norm
svm=cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_RBF)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)
imgs = []
def hog_single(img):
        samples=[]
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n*ang/(2*np.pi))
        bin_cells = bin[:100,:100], bin[100:,:100], bin[:100,100:], bin[100:,100:]
        mag_cells = mag[:100,:100], mag[100:,:100], mag[:100,100:], mag[100:,100:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps
        samples.append(hist)
        return np.float32(samples)
def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n*ang/(2*np.pi))
        bin_cells = bin[:100,:100], bin[100:,:100], bin[:100,100:], bin[100:,100:]
        mag_cells = mag[:100,:100], mag[100:,:100], mag[:100,100:], mag[100:,100:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps
        samples.append(hist)
    return np.float32(samples)
def train_svm(num):
    for img in glob.glob(r"C:\Users\NADEEM\Desktop\dataset\*.jpg"):
        n = cv2.imread(img, 0)
        imgs.append(n)
    labels = np.repeat(np.arange(1, num+1), 3200) # label for each corresponding image saved above repeat the elements of the array for 400 times
    samples = preprocess_hog(imgs)
    print(len(labels))
    print(len(samples))
    svm.train(samples,cv2.ml.ROW_SAMPLE,labels)
    return svm
def predict(svm,img):
     samples1=hog_single(img)
     resp=svm.predict(samples1)
# print(type(resp))
     return np.asscalar((resp[1].ravel().astype(int)))


# predict(Model,img[0])
if cv2.waitKey(1) & 0xFF == ord('q'):
  cv2.destroyAllWindows()