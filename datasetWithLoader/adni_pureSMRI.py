from typing import Optional, Sequence, Union
from monai.transforms import (Compose,LoadImaged,EnsureChannelFirstd,Orientationd,Resized, Spacingd,ScaleIntensityRanged,CropForegroundd,SaveImaged,NormalizeIntensityd,
                                ScaleIntensityd,RandRotate90d, CenterSpatialCropd, ScaleIntensityRangePercentilesd, NormalizeIntensityd, CopyItemsd)
from monai.data import DataLoader, ImageDataset, FolderLayout

# 为了暂时测试
from monai.data import CacheDataset
# from monai.data import Dataset as CacheDataset

import json
import os
import pandas as pd

# 主要的目的就是返回处理好的图片数据

# 一、生成函数
# 函数1：功能：生成add的数据
#       返回：没有返回值，在传入的train列表中拼接
#           public_modality + add_train_modality 中的n1 + n2个样本 
#           后面的n2个样本，其中缺失的模态，需要根据n1中对应的类别生成（提供选项来保存或者丢弃）

# 二、返回给dataloader的调用者
# 函数2：功能：划分为train、test（目前不用val）
#       返回：public_modality的n1个样本，每个样本中每一种图片模态的信息经过了不同的预处理，label就保持原样返回

        # 接口定义
                # 传入：jsonPath:"str", askListName:Optional["train_list", "test_list"], addTrainDataFlag: Optional[True, False]=False
                # 返回：train_dataLoader、test_dataLoader
class ADNIlist:

    def __init__(self, jsonPath:str):
        
        self.jsonPath = jsonPath

    # 提取json中的list
    def _getList(self, askListName: str): 

        # 训练集，打开JSON文件并读取数据
        with open(self.jsonPath, 'r') as file:
            json_data = json.load(file)
        
        # 提取img_path和img_label
        data_list = json_data.get(askListName, [])  # 获取data_list字段，如果不存在则返回空列表

        # test_df = pd.DataFrame(data_list)
        # # 这样一个括号，就能变成元组，进行两个数据的打包
        # test_df.to_dict(orient='records')

        return data_list

    def _getTest_DataSet(self, MiddleSize=(96, 96, 96)):

        test_transforms = Compose([

                            CopyItemsd(keys=["sMRI"], times=1, names=["path"], allow_missing_keys=True),

                            # 1、加载图像
                            LoadImaged(keys=["sMRI"],ensure_channel_first=True,allow_missing_keys=True),

                            # 2、确保坐标对
                            Orientationd(keys=["sMRI"],axcodes="RAS", allow_missing_keys=True),

                            # # 3、调整尺寸
                            Resized(keys="sMRI",spatial_size = (96, 96, 96), allow_missing_keys=True),
                            # Resized(keys="sMRI",spatial_size = (192, 192, 192), allow_missing_keys=True),

                            # Resized(keys="sMRI",spatial_size = MiddleSize, allow_missing_keys=True),
                            # CenterSpatialCropd(keys="sMRI",roi_size= [96, 96, 96], allow_missing_keys=True),

                            ScaleIntensityd(keys=["sMRI"], minv=0,maxv=128,  allow_missing_keys=True),

                            # 4、将所有的体素值强度归一化到[0,128]之间
                            # NormalizeIntensityd(keys=["sMRI"], nonzero=True),
                                                        
                            # SaveImaged(keys=["sMRI","fMRI-fALFF"],folder_layout=name_method)

                        ],map_items=True)

        # 2、设置好需要加载数据的集合set
        PARENT_FOLDER = self._getList(askListName="metainfo")["parent_folder"][0]
        public_modality = self._getList(askListName="metainfo")["public_modality"]

        test_list = self._getList(askListName="test_list")
        testDataFrame = pd.DataFrame(test_list)

        # 遍历DataFrame的所有列，为剩下的列的项都加上前缀
        for column_name in public_modality:
            testDataFrame[column_name] = [os.path.join(PARENT_FOLDER, item) for item in testDataFrame[column_name].astype(str)]
        
        imgDictList = testDataFrame.to_dict(orient="records")

        # current_dataset = Dataset(data=imgDictList, transform=test_transforms)
        current_dataset = CacheDataset(data=imgDictList, transform=test_transforms)

        return current_dataset

    # 根据2 要求的是public的还是addTrainDataFlag的值
    # 返回 适合于处理list中每一项数据的transforms，其中每项都是字典类型
    def _getTrain_DataSet(self, TargetFolder="./TEMP_transForms", addTrainDataFlag:bool = False, MiddleSize=(96, 96, 96)):

        # 1、设置好trandforms
        if not os.path.exists(TargetFolder):
            os.makedirs(TargetFolder)
        name_method = FolderLayout(
                    output_dir=TargetFolder,
                    extension="nii.gz",
                    makedirs=False)
        name_method.filename(subject="")

        train_transforms = Compose([  
                            # 1、加载图像
                            LoadImaged(keys=["sMRI"],ensure_channel_first=True,allow_missing_keys=True),

                            # 2、确保坐标对
                            Orientationd(keys=["sMRI"],axcodes="RAS",allow_missing_keys=True),

                            # # 3、调整尺寸
                            Resized(keys="sMRI",spatial_size = (96, 96, 96),allow_missing_keys=True),
                            # Resized(keys="sMRI",spatial_size = (96, 96, 96), allow_missing_keys=True),

                            # Resized(keys="sMRI",spatial_size = MiddleSize, allow_missing_keys=True),
                            # CenterSpatialCropd(keys="sMRI",roi_size= [96, 96, 96], allow_missing_keys=True),

                            ScaleIntensityd(minv=0,maxv=128, keys=["sMRI"], allow_missing_keys=True),

                            # 4、将所有的体素值强度归一化到[0,128]之间
                            # NormalizeIntensityd(keys=["sMRI"], nonzero=True),
                            
                            # RandRotate90d(keys=["sMRI"], allow_missing_keys=True)

                            # SaveImaged(keys=["sMRI","fMRI-fALFF"],folder_layout=name_method)

                        ],map_items=True)

        # 2、设置好需要加载数据的集合set
        PARENT_FOLDER = self._getList(askListName="metainfo")["parent_folder"][0]
        public_modality = self._getList(askListName="metainfo")["public_modality"]

        train_list = self._getList(askListName="train_list")
        trainDataFrame = pd.DataFrame(train_list)

        # label_list = trainDataFrame['label'].to_list()  # 由于下面想在原地修改，如果需要复制dataframe的一行，就必须使用copy()
        # trainDataFrame.drop('label', axis=1, inplace=True)

        # 遍历DataFrame的所有列，为剩下的列的项都加上前缀
        # for column_name in trainDataFrame.columns:
        for column_name in public_modality:
            trainDataFrame[column_name] = [os.path.join(PARENT_FOLDER, item) for item in trainDataFrame[column_name].astype(str)]
        
        imgDictList = trainDataFrame.to_dict(orient="records")

        # current_dataset = Dataset(data=imgDictList, transform=train_transforms)
        current_dataset = CacheDataset(data=imgDictList, transform=train_transforms)

        return current_dataset

    # 根据1 run_stage处于train、test的不同阶段
    def getDataLoader(self, run_stage: str,
                      batch_size: int = 1, shuffle:bool=False, num_workers:int = 0 , pin_memory:bool=True, MiddleSize=[96, 96, 96]): 

        if run_stage=="train":
            current_dataset = self._getTrain_DataSet(MiddleSize=MiddleSize)
        elif run_stage=="train_val":
            current_dataset = self._getTest_DataSet(MiddleSize=MiddleSize)
        elif run_stage=="test":
            current_dataset = self._getTest_DataSet(MiddleSize=MiddleSize)

        # 寻找指定模态的数据文件夹，要求和json路径中的annotations在同一级目录
        loader = DataLoader(
                batch_size=batch_size,
                shuffle=shuffle,
                dataset=current_dataset,
                num_workers=num_workers,
                pin_memory=pin_memory,
                # collate_fn=pad_list_data_collate
            )

        return loader
    
    # # 必须先实例化self.dataset
    # def get_image_files(self):

    #     assert self.dataset is not None,f"必须先调用getLoader方法来实例化dataset"
    #     return self.dataset.image_files

    # def get_dataSetLen(self):

    #     return len(self.dataset)

# train_loader = ADNIlist(jsonPath="/my_data/dataset/ADNI/5_link_dataset/pureSMRI_oneImgForOneSubject/multiClass/fiveFold_1.json").getDataLoader(run_stage="train")
# for batch in train_loader:
#     test_vector = batch
#     print('hello')

# test_loader = ADNIlist(jsonPath="/my_data/dataset/ADNI/5_link_dataset/fiveFold_1.json").getDataLoader(run_stage="test")
# for batch in test_loader:
#     test_vector = batch
#     print('hello')