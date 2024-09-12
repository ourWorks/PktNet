import os
import numpy as np
from monai.transforms import (Compose,LoadImaged,EnsureChannelFirstd,Orientationd,Resized, Spacingd,ScaleIntensityRanged,CropForegroundd,SaveImaged,NormalizeIntensityd)
import subprocess

import sys
sys.path.append('/my_data/project/python/classification/end/preprocess')  ###【十分关键，这个添加最大的项目文件夹下面的相对路径就可以了】
from my_transform.RemoveEdged_transform import RemoveEdged
from my_transform.RemoveTwoEdged_transform import RemoveTwoEdged

TEMPLATE_PATH = "/my_data/dataset/template/MNI152_T1_1mm_brain.nii.gz"
ATALS_PATH = "/my_data/dataset/template/AAL3v1_1mm.nii.gz"

# # 测试专用
# MODALITY_PATH = "/my_data/dataset/ADNI/2_choose_need_images/test_T1_Transform/T1_1_NII"
# REORIENT_PATH = "/my_data/dataset/ADNI/2_choose_need_images/test_T1_Transform/T1_2_RRORIENT"
# BET_PATH = "/my_data/dataset/ADNI/2_choose_need_images/test_T1_Transform/T1_3_BET"
# ORAL_BRAIN_PATH = "/my_data/dataset/ADNI/2_choose_need_images/test_T1_Transform/T1_4_ORAL_BRAIN"
# MNI_BRAIN_PATH = "/my_data/dataset/ADNI/2_choose_need_images/test_T1_Transform/T1_5_MNI152_BRAIN"
# SEG_PATH = "/my_data/dataset/ADNI/2_choose_need_images/test_T1_Transform/T1_6_BRAIN_SEG"

########################################################################################################
# 已执行过的大批数据

MODALITY_PATH = "/my_data/dataset/ADNI_1207/2_choose_need_images/sMRI/T1_1_NII"
REORIENT_PATH = "/my_data/dataset/ADNI_1207/2_choose_need_images/sMRI/T1_2_RRORIENT"
BET_PATH = "/my_data/dataset/ADNI_1207/2_choose_need_images/sMRI/T1_3_BET"
ORAL_BRAIN_PATH = "/my_data/dataset/ADNI_1207/2_choose_need_images/sMRI/T1_4_ORAL_BRAIN"
MNI_BRAIN_PATH = "/my_data/dataset/ADNI_1207/2_choose_need_images/sMRI/T1_5_MNI152_BRAIN"
SEG_PATH = "/my_data/dataset/ADNI_1207/2_choose_need_images/sMRI/T1_6_BRAIN_SEG"


folders = [MODALITY_PATH, REORIENT_PATH, BET_PATH, ORAL_BRAIN_PATH, MNI_BRAIN_PATH, SEG_PATH]
for folder in folders:
    # if os.path.exists(folder):
    os.makedirs(folder, exist_ok=True)  # exist_ok=True就是说，如果存在的话就不重新创建了，原有的文件也会保留；如果不设置就默认为False，会报错

# 定义自定义的滤波函数
def threshold_at_one(x):
    # threshold at 1
    return x < 390

# 定义多线程的设置：如果某些操作消耗内存差距很大，就再每个操作的前后分别设置好max_connections的数量
import threading
max_connections = 14  # 定义最大线程数
pool_sema = threading.BoundedSemaphore(max_connections)  # 或使用Semaphore方法

# 一、加载图像
# !!!SaveImaged不要将后缀前的文件名设置为seg，否则3DSlicer会将这个文件看作分割的label，然后会按照数值来显示为标签！！！

from monai.data import FolderLayout
def getRemoveLoader(TargetFolder,newPostFixWithoutLine, edgeNum):

    if not os.path.exists(TargetFolder):
        os.makedirs(TargetFolder)
    name_method = FolderLayout(
                output_dir=TargetFolder,
                # postfix="seg",
                extension="nii.gz",
                # parent=True,
                postfix=newPostFixWithoutLine,
                makedirs=False)
    name_method.filename(subject="")

    # name_method.filename(idx="stageNameWithoutLine", modality="T1")

    return  Compose([  # 1、加载图像
                        LoadImaged(keys="image",ensure_channel_first=True),

                        # 2、确保坐标对
                        Orientationd(keys="image",axcodes="RAS"),

                        # 3、自定义滤波加【周边收缩】，裁减边界区域
                        RemoveEdged(keys="image",edgeNum = edgeNum),
                        EnsureChannelFirstd(keys="image"),

                        # ScaleIntensityRanged(keys=("image"),a_min=390, a_max=1000, clip=True), # 这个处理是用来去掉极端大、极端小的数值

                        # 4、根据体素值强度进行范围缩放、根据设置的阈值进行裁减
                        # ScaleIntensityd(keys=("image")), # 先将所有体素值缩放到一个范围
                        # CropForegroundd(keys=("image"), source_key="image", select_fn=threshold_at_one, margin=0, return_coords=True), # 设置裁剪信息

                        # KeepLargestConnectedComponentd(keys=("image"), connectivity=3, independent=False), # connectivity是为了说明邻接的情况,independent默认为True就是按数值来划分
                        
                        # SaveImaged(keys="image",separate_folder=True, output_ext="nii.gz",output_postfix=stageNameWithoutLine,
                        #             output_dir=MODALITY_PATH),
                        SaveImaged(keys="image",folder_layout=name_method),
                    ],map_items=True)
                
                # map_items定位于monai.transforms.transform.py line:139行，如果按照默认的是true，就没有办法单独便利某个项    
                # if isinstance(data, (list, tuple)) and map_items:

def useAALModel(TargetFolder,newPostFixWithoutLine, edgeNum):

    if not os.path.exists(TargetFolder):
        os.makedirs(TargetFolder)
    name_method = FolderLayout(
                output_dir=TargetFolder,
                extension="nii.gz",
                postfix=newPostFixWithoutLine,
                makedirs=False)
    name_method.filename(subject="")

    # name_method.filename(idx="stageNameWithoutLine", modality="T1")

    return  Compose([  
                        # 1、加载图像
                        LoadImaged(keys=("image","label"),ensure_channel_first=True),

                        # 2、确保坐标对
                        Orientationd(keys=("image","label"),axcodes="RAS"),

                        # 3、双裁减，模型+aal模板：和removeEdge的区别是
                        # （1）RemoveEdge是MRI和label分别独立进行裁减
                        # （2）RemoveTwoEdged是根据MRI进行裁减，然后label也是根据MRI的裁减信息进行裁减
                        RemoveTwoEdged(keys=("image","label"),edgeNum = edgeNum),

                        # 4、调整尺寸:下面的尺寸是估算出来的:peizhunzhiqian  # 不调整的时候，大概是(108, 168, 138)
                        # Resized(keys=("image","label"),spatial_size = (96, 96, 96)),

                        # 5、将所有的体素值强度归一化到[0,1]之间
                        # ScaleIntensityd(minv=0,maxv=256,keys=("image","label")),

                        SaveImaged(keys=("image","label"),folder_layout=name_method),
                    ],map_items=True)


# # 一、【对齐方向】###############################################################################
reorientProfix = "_reorient.nii.gz"

# for orientItem in os.listdir(MODALITY_PATH):
def oneOrient(idx, orientItem):
    sourcePath = os.path.join(MODALITY_PATH, orientItem)
    targetBaseName = orientItem.replace(".nii.gz", reorientProfix)
    targetPath = os.path.join(REORIENT_PATH, targetBaseName)
    reorient_cmd = ["fslreorient2std", sourcePath, targetPath]

    pool_sema.acquire()  # 线程准备开始，加锁，限制线程数
    print('assignment %s start,item %s' % (idx, orientItem))
    if not os.path.exists(targetPath):
        subprocess.run(reorient_cmd)
    print('assignment %s end,item %s' % (idx, item))
    pool_sema.release()  # 线程结束，解锁

# for orientItem in os.listdir(MODALITY_PATH):
#     oneOrient(orientItem)
ImgList = os.listdir(MODALITY_PATH)
# 创建线程列表
threads = []
ImgList.sort()

# 循环一：里面没有thread.join()，主线程可以继续向下执行，不必等待每个线程都启动
for idx,item in enumerate(ImgList, start=1):
    thread = threading.Thread(target=oneOrient, args=(idx, item))
    threads.append(thread)
    thread.start()

# 主线程等待所有子线程执行完毕
for thread in threads:
    thread.join() #【必须加上】

# # 二、【分割BET】###############################################################################

## 【直接使用文件夹会更快，因为某些权重可以只加载一次：这里要求后面的代码适应它输出的名称
bet_cmd = ["hd-bet", "-i", REORIENT_PATH, "-o", BET_PATH]
# 只有当后面的文件夹文件数量比较小的时候，才运行下面这条命令
if len(os.listdir(REORIENT_PATH)) > len(os.listdir(BET_PATH)):
    subprocess.run(bet_cmd)


# # 三、【收缩边界区域、调整脑部区域的尺寸】###################################################################################
# 【看清楚，这次目标文件夹换了】
############################不变的预先设置代码
reorientProfix = "_reorient.nii.gz"
removeEdgeProfix = "RemoveEdge" # 用于monai的后缀输入，不需要写清下划线和后缀

imgDictList = [{"image":os.path.join(BET_PATH,item)} for item in os.listdir(BET_PATH) if item.endswith(reorientProfix)] # 这次的后缀和前面的前面一样
removeEdge_loaderOne = getRemoveLoader(TargetFolder=BET_PATH, newPostFixWithoutLine=removeEdgeProfix, edgeNum=20)
flag = False if len([item for item in os.listdir(BET_PATH) if item.endswith(f"{removeEdgeProfix}.nii.gz")]) > 0 else True
if flag:
    for short_list in [imgDictList[i:i + 300] for i in range(0, len(imgDictList), 300)]:
        removeEdge_loaderOne(short_list)


# 四、创建brain的软链接##############################################################################################
removeEdgeProfix = "RemoveEdge"
oralBrainProfix = "_oralBrain.nii.gz"

imgAbsPathList = [os.path.join(BET_PATH,item) for item in os.listdir(BET_PATH) if item.endswith(f"{removeEdgeProfix}.nii.gz")]                                                       
for item in imgAbsPathList:
    sourceLink = item
    # 只需要处理后的brain
    newBaseNameWithoutSuffix = os.path.basename(item).split("_r")[0]
    newLink = os.path.join(ORAL_BRAIN_PATH, f"{newBaseNameWithoutSuffix}{oralBrainProfix}")
    if not os.path.exists(newLink):
        os.symlink(sourceLink, newLink, target_is_directory=False)


# # 五、配准到MNI152模板##############################################################################################
mniBrainProfix = "_mniBrain"
def oneFlirt(idx, item):
    # 只需要处理后的brain
    newBaseNameWithoutSuffix = os.path.basename(item).replace(oralBrainProfix, mniBrainProfix)
    newNiiPath = os.path.join(MNI_BRAIN_PATH, f"{newBaseNameWithoutSuffix}.nii.gz")
    newMatPath = os.path.join(MNI_BRAIN_PATH, f"{newBaseNameWithoutSuffix}_affine.mat")
    cmd = ["flirt", "-in", item, "-ref", TEMPLATE_PATH,  "-out", newNiiPath, "-omat", newMatPath]

    pool_sema.acquire()  # 线程准备开始，加锁，限制线程数
    print('assignment %s start,item %s' % (idx, item))
    #######################################################【不存在才生成】
    if not os.path.exists(newNiiPath):
        subprocess.run(cmd)

    print('assignment %s end,item %s' % (idx, item))
    pool_sema.release()  # 线程结束，解锁

imgList = [os.path.join(ORAL_BRAIN_PATH,item) for item in os.listdir(ORAL_BRAIN_PATH)] # oralXxx文件夹里面就只有一种类型的文件
# 创建线程列表
threads = []
imgList.sort()
for idx,item in enumerate(imgList, start=1):
    thread = threading.Thread(target=oneFlirt, args=(idx, item))
    threads.append(thread)
    thread.start()

# 主线程等待所有子线程执行完毕
for thread in threads:
    thread.join() #【必须加上】

# 六、和AAL模板对准，只裁减出脑部区域############################################################################################ 缩小前大约的尺寸：(171, 160, 141)

mniBrainProfix = "_mniBrain"
tempLabelFolder = os.path.join(os.path.dirname(ATALS_PATH), "temp")
os.makedirs(tempLabelFolder, exist_ok=True)

imgDictList = []
for item in os.listdir(MNI_BRAIN_PATH):
    if item.endswith(f"{mniBrainProfix}.nii.gz"):
        
        # 重新命名atals模板文件
        targetAtalsName = os.path.join(tempLabelFolder, item.replace(".nii","_label.nii"))
        if not os.path.exists(targetAtalsName):
            os.symlink(ATALS_PATH, targetAtalsName, target_is_directory=False)

        one_dict = {"image":os.path.join(MNI_BRAIN_PATH,item),"label":targetAtalsName}

        imgDictList.append(one_dict)
        
removeEdge_cropLoader = useAALModel(TargetFolder=MNI_BRAIN_PATH, newPostFixWithoutLine="removeEdge", edgeNum=0)
# for short_list in [imgDictList[i:i + 200] for i in range(0, len(imgDictList), 200)]:
#     removeEdge_cropLoader(short_list)
for item in imgDictList:
    finalName = os.path.join(MNI_BRAIN_PATH, os.path.basename(item["image"]).replace(".nii.gz","_removeEdge.nii.gz"))
    if not os.path.exists(finalName):
        removeEdge_cropLoader(item)
