#by Qianqian Jiang
#20230413
#对FY3D进行大气校正
import h5py
import numpy as np
import time
from read_FY3D import read_spec_info,read_rad_info
from AOD_func_FY3D import get_imgAOD,Read_LUT
from osgeo import gdal

# def cal_surf_img(TOA_Ref,aod_img,surface_ref_sw,Ref_LUT_Band, Trans_LUT_Band, Sph_albedo_LUT_Band, AOD_LUT_1, AOD_LUT_2,
#         para_AOD,):


if __name__=='__main__':
    t1 = time.time()
    inpath = r"F:\jqq\高光谱项目\ac_code\ac_code\FY-3D\DATA\FY3D\hdf\2"
    flag=1 #flag=0是250m,flag=1是1km
    # 读取影像辐射信息
    TOA_Ref_img, SZA_img, VZA_img, Lat_img, Long_img, CLM_img, date,geopath = read_rad_info(inpath, flag)
    TOA_Ref_img=TOA_Ref_img/10000
    # 读取影像光谱信息
    fltbpath = r"F:\jqq\高光谱项目\ac_code\ac_code\FY-3D\DATA\rtcoef_fy3_4_mersi2_srf"
    Spec, Center_Wave = read_spec_info(fltbpath)
    t2=time.time()
    print("读取影像信息用时：%.2fs"%(t2-t1))
    if flag==0:
        band_num=4
    else:
        band_num=7
    TOA_Ref_img=TOA_Ref_img[:band_num,:,:]
    Row = TOA_Ref_img.shape[1]
    Col = TOA_Ref_img.shape[2]
    MOD=1
    #反演AOD
    AOD_img, flag = get_imgAOD(TOA_Ref_img, Lat_img, Long_img, date, CLM_img, SZA_img, VZA_img,Row,Col,band_num)
    AOD_img = np.array(AOD_img)/1000
    t3=time.time()
    print("反演AOD用时：%.2f"%(t3-t2))
    SURF=np.zeros(TOA_Ref_img.shape)
    if flag:
        #进行大气校正
        #读取查找表
        ###############################对整景影像分幅进行计算，分成1行5列################################
        n = 10
        width = int(Col / n)
        for i in range(n):
            # i=4
            start_col = i * width
            end_col = min((i + 1) * width, Col)

            Ref_part=TOA_Ref_img[:,:,start_col:end_col]
            row=Ref_part.shape[1]
            col=Ref_part.shape[2]
            AOD_img_part=AOD_img[:,start_col:end_col]

            SolarZenith = SZA_img[:, start_col:end_col]
            ViewZenith = VZA_img[:, start_col:end_col]
            meanSZA = np.median(SolarZenith)
            meanVZA = np.median(ViewZenith)
            AOD_List = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00, 1.20, 1.40, 1.60, 1.80, 2.00]
            Trans_LUT_List330 = []
            Ref_LUT_List330 = []
            Sph_albedo_LUT_List330 = []
            # 读取 VIS LUT列表（GF5B对应LUT)
            for aod in AOD_List:
                Trans_LUT, Ref_LUT, Sph_albedo_LUT = Read_LUT(MOD, meanSZA, meanVZA, aod)
                Trans_LUT_List330.append(Trans_LUT)
                Ref_LUT_List330.append(Ref_LUT)
                Sph_albedo_LUT_List330.append(Sph_albedo_LUT)
            Trans_LUT_List330=np.array(Trans_LUT_List330)
            Ref_LUT_List330=np.array(Ref_LUT_List330)
            Sph_albedo_LUT_List330=np.array(Sph_albedo_LUT_List330)

            # 大气校正准备影像
            AOD_LUT_1 = np.zeros([row, col])
            for i in range(len(AOD_List)):
                # mask = np.where(VIS_PT > AOD_LIST[i])
                AOD_LUT_1[AOD_img_part < AOD_List[i]] = AOD_LUT_1[AOD_img_part < AOD_List[i]] + 1
            AOD_LUT_2 = AOD_LUT_1
            AOD_LUT_1 = AOD_LUT_2 - 1
            para_AOD = np.zeros([row, col])
            AOD_LUT_2[AOD_LUT_2 == len(AOD_List)] = AOD_LUT_1[AOD_LUT_2 == len(AOD_List)]
            AOD_LUT_1[AOD_LUT_2 == 0] = AOD_LUT_2[AOD_LUT_2 == 0]
            AOD_LUT_MAX = max(map(max, AOD_LUT_2))
            AOD_LUT_MIN = min(map(min, AOD_LUT_1))
            # para_AOD = (VIS_PT - AOD_LIST[AOD_LUT_1]) / (AOD_LIST[AOD_LUT_1 + 1] - AOD_LIST[AOD_LUT_1])
            for j in range(int(AOD_LUT_MIN), int(AOD_LUT_MAX)):
                if (AOD_LUT_1 == j).any() and (AOD_LUT_2 == j).any():
                    mask = np.where((AOD_LUT_1 == j) & (AOD_LUT_2 == j))
                    para_AOD[mask[0][:], mask[1][:]] = 1
                if (AOD_LUT_1 == j).any() and (AOD_LUT_2 == j + 1).any():
                    mask = np.where((AOD_LUT_1 == j) & (AOD_LUT_2 == j + 1))
                    VIS_Para_mat = (AOD_img_part[mask[0][:], mask[1][:]] - AOD_List[j]) / (AOD_List[j + 1] - AOD_List[j])
                    para_AOD[mask[0][:], mask[1][:]] = VIS_Para_mat
            surface_ref=[]
            for i in range(band_num):
                TOA_Ref = Ref_part[i,:,:]
                # 确定大气参数
                Ref_LUT_img = np.zeros([row, col, 2], dtype='float32')
                Trans_LUT_img = np.zeros([row, col, 2], dtype='float32')
                Sph_LUT_img = np.zeros([row, col, 2], dtype='float32')
                for j in range(int(AOD_LUT_MIN), int(AOD_LUT_MAX) + 1):
                    if (AOD_LUT_1 == j).any():
                        mask = np.where(AOD_LUT_1 == j)
                        Ref_LUT_img[mask[0][:], mask[1][:], 0] = Ref_LUT_List330[j,i]
                        Trans_LUT_img[mask[0][:], mask[1][:], 0] = Trans_LUT_List330[j,i]
                        Sph_LUT_img[mask[0][:], mask[1][:], 0] = Sph_albedo_LUT_List330[j,i]
                    if (AOD_LUT_2 == j).any():
                        mask = np.where(AOD_LUT_2 == j)
                        Ref_LUT_img[mask[0][:], mask[1][:], 1] = Ref_LUT_List330[j,i]
                        Trans_LUT_img[mask[0][:], mask[1][:], 1] = Trans_LUT_List330[j,i]
                        Sph_LUT_img[mask[0][:], mask[1][:], 1] = Sph_albedo_LUT_List330[j,i]

                ref_atom_pixel = (1 - para_AOD) * Ref_LUT_img[:, :, 0] + (para_AOD) * Ref_LUT_img[:, :, 1]
                Trans_pixel = (1 - para_AOD) * Trans_LUT_img[:, :, 0] + (para_AOD) * Trans_LUT_img[:, :, 1]
                sph_albedo_pixel = (1 - para_AOD) * Sph_LUT_img[:, :, 0] + (para_AOD) * Sph_LUT_img[:, :, 1]
                TOA_Ref_Block = (TOA_Ref - ref_atom_pixel) / Trans_pixel
                surface_ref.append(np.rint((TOA_Ref_Block * (1 + TOA_Ref_Block * sph_albedo_pixel)) * 10000))
            surface_ref=np.array(surface_ref)
            SURF[:,:,start_col:end_col]=surface_ref/10000
        t4=time.time()
        print("大气校正用时：%.2f"%(t4-t3))
        SURF[:,CLM_img==1]=TOA_Ref_img[:,CLM_img==1]
        SURF=SURF*10000
        outfile = geopath.replace('GEO1K', '1000M').replace('.HDF', '_sr1.tiff')
        driver = gdal.GetDriverByName('GTiff')
        outraster = driver.Create(outfile, Col, Row, band_num, gdal.GDT_Int16)
        for i in range(band_num):
            outband = outraster.GetRasterBand(i+1).WriteArray(SURF[i,: ,:])
        outraster.FlushCache()

