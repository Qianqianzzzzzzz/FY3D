import os
import numpy as np
from read_FY3D import read_spec_info
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#通过地物光谱库分别和FY3D以及MODIS光谱响应函数积分，计算两者之间的差异。


#读取FY3D光谱响应函数
fltbpath = r"F:\jqq\高光谱项目\ac_code\ac_code\FY-3D\DATA\rtcoef_fy3_4_mersi2_srf"
Spec_FY, Center_Wave_FY = read_spec_info(fltbpath)

#读取MODIS第三波段光谱响应函数
MODIS_wavelength = np.arange(390,2201,1)
Spec_MODIS = np.zeros(2200-390+1)
filepath=r"F:\jqq\高光谱项目\GF5B-TOA验证\MODIS_SPEC.txt"
bandwave=[]
specsponse=[]
if os.path.exists(filepath):
    FILE_HANDLER = open(filepath, encoding="utf-8")
    dataList = FILE_HANDLER.readlines()
    for line in dataList[38:]:
        bandwave.append(float(line.split()[0]))
        specsponse.append(float(line.split()[3]))
    FILE_HANDLER.close()
bandwave=np.array(bandwave)
specsponse=np.array(specsponse)
for i in range(MODIS_wavelength.shape[0]):
    wave_dist=np.abs(bandwave-MODIS_wavelength[i]/1000)
    id=wave_dist.argsort()
    Spec_MODIS[i]=specsponse[id[0]]

# x=np.arange(390,2201,1)
# plt.plot(x,Spec_MODIS,label='MODIS',color='g')
# plt.plot(x,Spec_FY[:,0],label='FY3D',color='r')
# plt.xlim(380,600)
# plt.ylim(0,1.1)
# plt.xlabel('wavelength(nm)')
# plt.ylabel('SSF')
# plt.legend()
# plt.show()

#读取样本光谱库
pathDir = os.listdir(r'F:\jqq\材料-思琪\software\code\ac_code\Data\spec_basedata\new1')  # 获取当前路径下的文件名，返回list
sample_num = len(pathDir)
speclib_FY = np.zeros([len(pathDir)])
speclib_MODIS = np.zeros([len(pathDir)])
FILE = 0
for s in pathDir:
    newDir = os.path.join(r'F:\jqq\材料-思琪\software\code\ac_code\Data\spec_basedata\new1', s)  # 将文件名写入到当前文件路径后面
    if os.path.isfile(newDir):  # 如果是文件
        if s[-8:] == "AREF.txt":  # 判断是否是txt
            FILE_HANDLER = open(newDir, encoding="utf-8")
            # print(newDir)
            dataList = FILE_HANDLER.readlines()
            ref_spec = np.zeros(2200 - 390 + 1)  # 390-2200
            N = 0
            for line in dataList[41:]:
                ref_spec[N] = float(line.split('\t')[0])
                if ref_spec[N] <= 0:
                    ref_spec[N] = 0
                N = N + 1
                if N>=2200-390+1:
                    break
            speclib_FY[FILE] = np.sum(ref_spec * Spec_FY[:, 0]) / (np.sum(Spec_FY[:, 0]))
            speclib_MODIS[FILE] = np.sum(ref_spec * Spec_MODIS[:]) / (np.sum(Spec_MODIS[:]))

            FILE = FILE + 1
speclib_FY=speclib_FY[0:FILE]
speclib_MODIS=speclib_MODIS[0:FILE]
X = np.array(speclib_MODIS).reshape(-1, 1)
Y = np.array(speclib_FY).reshape(-1, 1)
reg = LinearRegression()
reg.fit(X, Y)
r2 = reg.score(X, Y)
a = reg.coef_[0]
b = reg.intercept_
plt.scatter(speclib_FY,speclib_MODIS,s=5)
x=np.linspace(0,1,100)
y=x
plt.plot(x,y,'b')
y=a*x+b
plt.plot(x,y,'r')
plt.text(0.05, 0.60, 'R\u00b2 = %.4f' % r2)
if b > 0:
    plt.text(0.05, 0.64, 'y=%.4f * x + %.4f' % (a, b))
else:
    plt.text(0.05, 0.64, 'y=%.4f * x - %.4f' % (a, abs(b)))
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel('MODIS_BAND3')
plt.ylabel('FY3D_Band1')
plt.show()