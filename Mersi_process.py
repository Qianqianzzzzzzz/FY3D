# Mersi L1数据的辐射定标
import h5py
import numpy as np
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import pandas as pd

# =====================================================================
# 相对方位角
# 相对方位角 = 太阳方位角 - 观测方位角
# 相对方位角:①0-180和180-360是对称的 所以一般也把角度归算到0-180的范围内 ②小于0 就加360
def cal_RelativeAzimuth(SolarAzimuth,SensorAzimuth):
    RA = SolarAzimuth - SensorAzimuth
    if RA < 0:
        RA = RA + 360
    if RA > 180:
        RA = 360 - RA
    return RA


# =====================================================================
# 辐射定标
# 计算公式
# dn=DN*Slope +Intercept 给定的官方文档说明了 反射通道中Slope值为1 Intercept值为0 所以这里省略该步骤
# Ref=Cal_2*dn2+Cal_1*dn+Cal_0
# 输入参数
# Cal：定标系数  HDF5 dataset
# DN：维度与定标系数的维度对应（注意不是一致是对应） HDF5 dataset
# sza：太阳天顶角
# 对于MersiⅡ 1000m Cal:shape(19,3) ----> DN:shape(19,c,r)
def calibration(Cal,DN,sza):
    DN2 = np.square(DN)
    Ref = np.ones(DN.shape)
    # 按通道分别进行计算
    for i in range(Cal.shape[0]):
        Ref[i,:,:] = (Cal[i,2] * DN2[i,:,:] + Cal[i,1] * DN[i,:,:] + Cal[i,0])/(100*np.cos(np.radians(sza)))
    return Ref

# =====================================================================
# 搜索某一站点对应的像元位置
# lon：经度
# lat：纬度
# Longitude：经度图像
# Latitude：纬度图像
def flux_loc(lon,lat,Longitude,Latitude):
    lon1, lat1, lon2, lat2 = np.radians(lon),np.radians(lat),np.radians(Longitude),np.radians(Latitude)
    # haversine公式
    d_lon = lon2 - lon1
    d_lat = lat2 - lat1
    aa = np.sin(d_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon / 2) ** 2
    dis = 2 * np.arcsin(np.sqrt(aa)) * 6371
    if np.min(dis) > 5:
        return None
    loc = np.argmin(dis)
    c = np.size(dis,1)
    row = loc // c
    col = loc % c
    return [row,col]


# =====================================================================
# 数据的批量处理（包括站点的筛选）
# 输入参数
# path：数据所在文件夹(包括地理文件 需要文件一一对应)
# path_clm：云检测产品所在文件夹
# flag：标志数据为1000m(1) 还是250m(0) 默认为250m
# fileout: 结果写入路径
def process_flux(path, fileout, path_clm, flux, flag=0):
    files = os.listdir(path)
    files_clm = os.listdir(path_clm)

    files_len = len(files)
    print("数据总数:",str(files_len))
    if (files_len % 2)==1:
        print("缺少地理文件或地理文件不对应")
        return 0

    # 批量读取并进行辐射定标处理
    # 注意 这里默认辐射文件和地理文件同时在一个文件夹内 且 一一对应 否则会报错
    for i in range(0,files_len,2):
        file_path = os.path.join(path,files[i])
        geofile_path = os.path.join(path,files[i+1])

        # 取出时间部分
        date = files[i][19:32]

        # 判断地理文件是不是一一对应
        if files[i][19:32]==files[i+1][19:32]:
            print("正在处理的文件时间：",date)
        else:
            print("缺少地理文件或文件不对应：",date)
            break

        # 搜索是否存在对应时间的云检测文件
        if [x for x in files_clm if x.find(date)!=-1]:
            clm_path = os.path.join(path_clm,[x for x in files_clm if x.find(date)!=-1][0])
            with h5py.File(clm_path, 'r') as chf:
                Cloud_Mask = np.array(chf['Cloud_Mask'])[0, :, :]
        else:
            print(date)
            print("缺少对应的云监测文件")
            continue

        # 由于250m和1000m数据文件hdf格式有区别这里 同时250m无自身的地理文件 需要进行插值处理 所以这里区分
        # 250m分辨率数据
        if flag==0:
            # 首先对云检测文件进行最邻近插值 注意其它插值方法不可行
            Cloud_Mask=scipy.ndimage.zoom(Cloud_Mask,4,order=0)
            with h5py.File(file_path,'r') as hf:
                # 包含了19个可见光通道的定标系数 只取前4个通道
                VIS_Cal_Coeff = np.array(hf['Calibration']['VIS_Cal_Coeff'][0:4,:])
                # 250m数据文件的通道较少 这里直接拼接成数组进行处理
                EV_Ref = np.array([hf['Data']['EV_250_RefSB_b1'],hf['Data']['EV_250_RefSB_b2'],hf['Data']['EV_250_RefSB_b3'],hf['Data']['EV_250_RefSB_b4']])
            with h5py.File(geofile_path,'r') as ghf:
                # 地理文件为1000m分辨率 对其进行插值
                # Bilinear interpolation would be order = 1,
                # nearest is order = 0,
                # and cubic is the default(order=3).
                Latitude = scipy.ndimage.zoom(ghf['Geolocation']['Latitude'], 4,order=1)
                Longitude = scipy.ndimage.zoom(ghf['Geolocation']['Longitude'], 4,order=1)
                SolarZenith = scipy.ndimage.zoom(ghf['Geolocation']['SolarZenith'], 4)*0.01  # 太阳天顶角用于计算反射率
                SolarAzimuth = scipy.ndimage.zoom(ghf['Geolocation']['SolarAzimuth'], 4)*0.01
                SensorZenith = scipy.ndimage.zoom(ghf['Geolocation']['SensorZenith'], 4)*0.01
                SensorAzimuth = scipy.ndimage.zoom(ghf['Geolocation']['SensorAzimuth'], 4)*0.01

        # 1000m分辨率数据
        elif flag==1:
            with h5py.File(file_path,'r') as hf:
                VIS_Cal_Coeff = np.array(hf['Calibration']['VIS_Cal_Coeff'])  # 定标参数
                EV_Ref = np.array([hf['Data']['EV_250_Aggr.1KM_RefSB'],hf['Data']['EV_1KM_RefSB']])         # 1000m反射通道dn值  250m反射通道dn值 聚合到1000m
            with h5py.File(geofile_path,'r') as ghf:
                Latitude = ghf['Geolocation']['Latitude']
                Longitude = ghf['Geolocation']['Longitude']
                SolarZenith = ghf['Geolocation']['SolarZenith']*0.01    # 太阳天顶角用于计算反射率
                SolarAzimuth = ghf['Geolocation']['SolarAzimuth']*0.01
                SensorZenith = ghf['Geolocation']['SensorZenith']*0.01
                SensorAzimuth = ghf['Geolocation']['SensorAzimuth']*0.01
        else:
            print("标识值错误：必须为0或1")
            return 0

        # 识别有云位置
        cloud_mask = (np.bitwise_and(Cloud_Mask,7))!=7
        cloud_mask = cloud_mask[::-1,::-1]
        # plt.imshow(cloud_mask)
        # plt.show()

        # 显示识别结果 与风云网站提供的影像对比 发现云检测影像需要翻转180度才能与数据文件对应
        # cloud_img = np.zeros(np.shape(cloud_mask))
        # cloud_img[cloud_mask] = 1
        # plt.imshow(cloud_img)
        # plt.show()

        # cloud_img2=cloud_img[::-1,::-1]
        # plt.imshow(cloud_img2)
        # plt.show()


        # 进行辐射定标
        Ref=calibration(VIS_Cal_Coeff,EV_Ref,SolarZenith)
        # 剔除有云数据
        # Ref[:,cloud_mask] = 0
        # plt.imshow(Ref[0, :, :])
        # plt.show()

        # 读取站点位置
        # 运行速度过慢 是否需要先提取 再进行相关处理
        for j in range(len(flux)):
            loc = flux_loc(flux.iloc[j,2], flux.iloc[j,1], Longitude, Latitude)
            # 不存在对应像素点
            if loc is None:
                continue
            r = Ref[:, loc[0], loc[1]]
            # 对应区域被云遮挡 无有效值
            if ~(np.any(r)) or r[0]<0 or r[1]<0 or r[2]<0 or r[3]<0:
                continue
            # 输出内容包括 时间 太阳天顶角 太阳方位角 观测天顶角 观测方位角 相对方位角 反射率 云检测数据(注意cloud_mask是以uint8形式存储的 需要转换)
            # 计算相对方位角
            alldata = pd.DataFrame([[date.replace('_',''),SolarZenith[loc[0], loc[1]],SolarAzimuth[loc[0], loc[1]],SensorZenith[loc[0], loc[1]],SensorAzimuth[loc[0], loc[1]],cal_RelativeAzimuth(SolarAzimuth[loc[0], loc[1]],SensorAzimuth[loc[0], loc[1]]),r[0],r[1],r[2],r[3], Cloud_Mask[loc[0], loc[1]].astype('float32') ]])
            alldata.to_csv(os.path.join(fileout,flux.iloc[j,0]+'.csv'),mode='a',index=False,header=False)



# =====================================================================
# 数据的批量处理（包括读取以及定标）
# 输入参数
# path：数据所在文件夹(包括地理文件 需要文件一一对应)
# path_clm：云检测产品所在文件夹
# flag：标志数据为1000m(1) 还是250m(0) 默认为250m
# fileout: 结果写入路径
def process(path, fileout, path_clm, flag=0):
    files = os.listdir(path)
    files_clm = os.listdir(path_clm)

    files_len = len(files)
    print("数据总数:",str(files_len))
    if (files_len % 2)==1:
        print("缺少地理文件或地理文件不对应")
        return 0

    # 批量读取并进行辐射定标处理
    for i in range(4,files_len,2):
        file_path = os.path.join(path,files[i])
        geofile_path = os.path.join(path,files[i+1])

        # 取出时间部分
        date = files[i][19:32]

        # 判断地理文件是不是一一对应
        if files[i][19:32]==files[i+1][19:32]:
            print("正在处理的文件时间：",date)
        else:
            print("缺少地理文件或文件不对应：",date)
            break
        # 搜索是否存在对应时间的云检测文件
        if [x for x in files_clm if x.find(date)!=-1]:
            clm_path = os.path.join(path_clm,[x for x in files_clm if x.find(date)!=-1][0])
            with h5py.File(clm_path, 'r') as chf:
                Cloud_Mask = np.array(chf['Cloud_Mask'])[0, :, :]
        else:
            print(date)
            print("缺少对应的云监测文件")
            continue

        # 由于250m和1000m数据文件hdf格式有区别这里 同时250m无自身的地理文件 需要进行插值处理 所以这里区分
        # 250m分辨率数据
        if flag==0:
            # 首先对云检测文件进行最邻近插值 注意其它插值方法不可行
            Cloud_Mask=scipy.ndimage.zoom(Cloud_Mask,4,order=0)
            with h5py.File(file_path,'r') as hf:
                # 包含了19个可见光通道的定标系数 只取前4个通道
                VIS_Cal_Coeff = np.array(hf['Calibration']['VIS_Cal_Coeff'][0:4,:])
                # 250m数据文件的通道较少 这里直接拼接成数组进行处理
                EV_Ref = np.array([hf['Data']['EV_250_RefSB_b1'],hf['Data']['EV_250_RefSB_b2'],hf['Data']['EV_250_RefSB_b3'],hf['Data']['EV_250_RefSB_b4']])
            with h5py.File(geofile_path,'r') as ghf:
                # 地理文件为1000m分辨率 对其进行插值
                # Bilinear interpolation would be order = 1,
                # nearest is order = 0,
                # and cubic is the default(order=3).
                Latitude = scipy.ndimage.zoom(ghf['Geolocation']['Latitude'], 4,order=1)
                Longitude = scipy.ndimage.zoom(ghf['Geolocation']['Longitude'], 4,order=1)
                SolarZenith = scipy.ndimage.zoom(ghf['Geolocation']['SolarZenith'], 4)*0.01  # 太阳天顶角用于计算反射率
        # 1000m分辨率数据
        elif flag==1:
            with h5py.File(file_path,'r') as hf:
                VIS_Cal_Coeff = np.array(hf['Calibration']['VIS_Cal_Coeff'])  # 定标参数
                EV_Ref = np.array([hf['Data']['EV_250_Aggr.1KM_RefSB'],hf['Data']['EV_1KM_RefSB']])         # 1000m反射通道dn值  250m反射通道dn值 聚合到1000m
            with h5py.File(geofile_path,'r') as ghf:
                Latitude = ghf['Geolocation']['Latitude']
                Longitude = ghf['Geolocation']['Longitude']
                SolarZenith = ghf['Geolocation']['SolarZenith']*0.01    # 太阳天顶角用于计算反射率
        else:
            print("标识值错误：必须为0或1")
            return 0

        # 识别有云位置
        cloud_mask = (np.bitwise_and(Cloud_Mask,7))!=7
        cloud_mask = cloud_mask[::-1,::-1]
        # plt.imshow(cloud_mask)
        # plt.show()

        # 显示识别结果 与风云网站提供的影像对比 发现云检测影像需要翻转180度才能与数据文件对应
        # cloud_img = np.zeros(np.shape(cloud_mask))
        # cloud_img[cloud_mask] = 1
        # plt.imshow(cloud_img)
        # plt.show()

        # cloud_img2=cloud_img[::-1,::-1]
        # plt.imshow(cloud_img2)
        # plt.show()


        # 进行辐射定标
        Ref=calibration(VIS_Cal_Coeff,EV_Ref,SolarZenith)
        # 剔除有云数据
        Ref[:,cloud_mask] = 0
        # plt.imshow(Ref[0, :, :])
        # plt.show()

        # 存储结果
        fileout_path = os.path.join(fileout,files[i])
        with h5py.File(fileout_path,'w') as ohf:
            geo = ohf.create_group("Geolocation")
            dt = ohf.create_group("Data")
            ohf.create_dataset('Geolocation/Latitude',data=Latitude)
            ohf.create_dataset('Geolocation/Longitude', data=Longitude)
            ohf.create_dataset('Geolocation/SolarZenith', data=SolarZenith)
            ohf.create_dataset('Data/EV_Ref', data=EV_Ref)
            ohf.create_dataset('Data/TOA_Ref', data=Ref)



# # =====================================================================
# # 同matlab同名函数
# def bitget(number, pos):
#     return (number >> pos) & 1
#
# def matrix2bin(m):
#     M=np.ones((8,)+m.shape)
#     for i in range(8):
#         M[i] = list(map(lambda x: (x >> i) & 1,m))
#     return M
#
# # 处理云检测文件
# def clm_process(path,  fileout):
#     files = os.listdir(path)
#     files_len = len(files)
#     print("云检测数据总数:",str(files_len))
#
#     # 批量读取并进行处理
#     for i in range(files_len):
#         file_path = os.path.join(path, files[i])
#         with h5py.File(file_path, 'r') as chf:
#             # chf.visit(print) # 查看文件格式
#             Cloud_Mask = np.array(chf['Cloud_Mask'])[0,:,:]
#         cloud = matrix2bin(Cloud_Mask)
#         pass



if __name__ == '__main__':
    # with h5py.File('F:/CAL/FY3D_MERSI_GBAL_L1_20200126_0110_0250M_MS.HDF','r') as f:
    # f.visit(print)

    # 数据所在路径
    path='I:/FY3D/250m/China/7'
    path_clm ='I:/FY3D/250m/China/clm'
    path_flux = 'I:/flux/heihe/2020/metadata.csv'
    flux = pd.read_csv(path_flux, names=['name','latitude','longitude','height','NO.'])

    process_flux(path, 'I:/MERSI_result/0315/heihe/7', path_clm,  flux)

    # process(path,'F:/CAL/',path_clm)

    # path_clm = 'F:/FY3D/2020/clm/'
    # clm_process(path_clm,0)




