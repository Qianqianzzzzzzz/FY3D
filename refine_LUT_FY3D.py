import numpy as np
import os
import math
from read_FY3D import read_spec_info
if __name__=='__main__':
    LUT_path=r"F:\jqq\高光谱项目\ac_code\ac_code\Data\Wavelength_LUT_aodmore.txt"
    f_LUT=open(LUT_path,"r")
    LUT=f_LUT.readlines()[2:]

    # 读取FY3D光谱响应函数
    fltbpath = r"F:\jqq\高光谱项目\ac_code\ac_code\FY-3D\DATA\rtcoef_fy3_4_mersi2_srf"
    Spec, Center_Wave = read_spec_info(fltbpath)

    LUT_GF5B_path=r"F:\jqq\高光谱项目\ac_code\ac_code\FY-3D\DATA\Wavelength_LUT_aodmore_FY3D.txt"
    f_LUT_GF5B=open(LUT_GF5B_path,"w")

    wave_id = (Center_Wave).argsort()
    Spec=Spec[:, wave_id]
    for i in range(int(len(LUT)/(2550-355+1+1))):
        temp=LUT[i*(2550-355+1+1):(i+1)*(2550-355+1+1)]
        f_LUT_GF5B.write(temp[0])
        Trans_LUT=[]
        Ref_LUT=[]
        Sph_albedo_LUT=[]
        Trans_LUT_cal=[]
        Ref_LUT_cal=[]
        Sph_albedo_LUT_cal=[]
        for j in temp[1:]:
            Trans_LUT.append(float(j.strip('\n').split('\t')[1]))
            Ref_LUT.append(float(j.strip('\n').split('\t')[2]))
            Sph_albedo_LUT.append(float(j.strip('\n').split('\t')[3]))
        Trans_LUT=np.array(Trans_LUT)
        Ref_LUT=np.array(Ref_LUT)
        Sph_albedo_LUT=np.array(Sph_albedo_LUT)
        Trans_LUT_cal=np.dot(Trans_LUT[390-355:390-355+1811].T, Spec) / (np.sum(Spec, axis=0))
        Ref_LUT_cal=np.dot(Ref_LUT[390-355:390-355+1811].T, Spec) / (np.sum(Spec, axis=0))
        Sph_albedo_LUT_cal=np.dot(Sph_albedo_LUT[390-355:390-355+1811].T, Spec) / (np.sum(Spec, axis=0))
        for k in range(len(Trans_LUT_cal)):
            f_LUT_GF5B.write(str(wave_id[k]+1)+"\t"+str(Trans_LUT_cal[k])+"\t"+str(Ref_LUT_cal[k])+"\t"+str(Sph_albedo_LUT_cal[k])+"\n")

    f_LUT_GF5B.close()
    f_LUT.close()
