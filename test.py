import numpy as npimport cv2img = cv2.imread('D:\\opencv\\sources\\samples\\python2\\data\\digits.png')
'''cv2.namedWindow("winname", 0)
cv2.imshow("winname", img)
cv2.waitKey(0)'''
# rgb->gray
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#把图片分隔成5000个，每个20x20大小
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]#再转成numpy数组
x = np.array(cells)#一半用来训练的数组，一半用来测试的数组
train = x[:,:50].reshape(-1,400).astype(np.float32)    # 提高数据精度
test = x[:,50:100].reshape(-1,400).astype(np.float32)#创建训练和测试的标签
k = np.arange(10)
train_labels = np.repeat(k,250)[:,np.newaxis]
test_labels = train_labels.copy()
from sklearn import svm
clf = svm.LinearSVC()
clf.fit(train, train_labels)
result = clf.predict(test)
result = result[:,np.newaxis]#最终检查测试的精确度，比较结果，检查哪些是错误的，最终输出正确率
matches = result == test_labelsprint(test.shape)
correct = np.count_nonzero(matches)
accuracy = correct*100.0 / result.sizeprint(accuracy)
