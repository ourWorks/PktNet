import dicom2nifti
import os
import shutil

def __dicom_to_nifti(modalityName, source_directory, target_directory):

    # root = "data/XX"  #root path is like 'data/XX'
    # converted_path = "nii_data/XX"  #path of converted_files_path to save
    # f_path = os.path.join(root,"fmri")# files of my fmri is like 'data/XX/fmri'
    # converted_f_path = os.path.join(converted_path,"fmri")#converted fmri files path

    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    count = 0
    total_length = len(os.listdir(source_directory))

    needItemList = []

    for item in os.listdir(source_directory):

        temp = item.split("-I")[0]
        final_itemName = os.path.join(target_directory, f"{modalityName}_{temp}.nii.gz")
        
        # 将需要的文件item存好
        needItemList.append(f"{modalityName}_{temp}.nii.gz")

        if os.path.exists(final_itemName):
            continue

        # 为了适应monai的命名模式需要将所有转换后的不同subject放到一个文件夹；但是这个dicom2nifti.convert_directory不支持更改文件名，只能在单独的文件夹中生成，然后再提取出来
        targetChildFolder = os.path.join(target_directory,item.split("-I")[0])

        # 1、耗时的转化部分
        sourceFolder = os.path.join(source_directory,item)# single fmri series directory path

        print(targetChildFolder)
        if not os.path.exists(targetChildFolder):
            os.makedirs(targetChildFolder)
            dicom2nifti.convert_directory(sourceFolder,output_folder=targetChildFolder) # 下面是等效写法
        else:
            print("Why")
        
        count+=1

        print(f"{count}/{total_length}...")

    # 生成的还在文件夹，那么文件中名字不符合，肯定要删除
    nii_Files = [file for file in os.listdir(target_directory) if os.path.isfile(os.path.join(target_directory, file))]
    for item in nii_Files:
        if item not in needItemList:
            print(f"{item}已经是多余的了")

# 先处理共有的文件夹名称：文件夹从1开始，然后再处理只有的文件夹名称；序号对应上
# 【健壮性测试】：
#   允许跳过目标文件夹已经存在的文件的【耗时的】转化步骤、
#   允许只保留source中的指定的样本文件、
#   支持和原有的文件进行重新排序，并加上idx序号

def extract_and_rename_files(source_directory, target_directory,rename_start_index):

    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    modalityName = os.path.basename(target_directory).split("_")[0] #从目标文件夹的baseName开头中获取modalityName

    # 【一】去除idx
    # 对之前生成过的单个文件去掉idx,但是以modalityName开头的就不要去掉前面的字符了
    filesListInTargetFolder = [item for item in os.listdir(target_directory) if not item.startswith(modalityName) and os.path.isfile(os.path.join(target_directory, item))]
    for item in filesListInTargetFolder:
        old_fileName = os.path.join(target_directory, item)
        new_fileName = os.path.join(target_directory, item[5:])
        os.rename(old_fileName, new_fileName)

    # targetFilesList = [item.replace("nii.gz","").replace(f"modalityName_","") for item in os.listdir(target_directory)]
    # # 遍历target_folder对不需要的文件进行删除

    # 【二】转换为nii文件
    ### 遍历source_directory转化，生成的nii文件在子文件夹里面
    for item in os.listdir(source_directory):

        temp = item.split("-I")[0]
        final_itemName = os.path.join(target_directory, f"{modalityName}_{temp}.nii.gz")
        if os.path.exists(final_itemName):  # 已经生成过的就不要再次生成了
            continue
        else:
            # 耗时的转化部分,且不能使用多线程
            sourceChildFolder = os.path.join(source_directory,item)# single fmri series directory path
            targetChildFolder = os.path.join(target_directory,item.split("-I")[0])
            os.makedirs(targetChildFolder)
            dicom2nifti.convert_directory(sourceChildFolder,output_folder=targetChildFolder) # 下面是等效写法
    
    # 【三】把样本文件夹里面的nii.gz文件取出来，并删除样本文件夹
    
    # 获取目标目录下的所有子文件夹【只需要文件夹】
    subdirectories = [os.path.join(target_directory,item) for item in os.listdir(target_directory) if os.path.isdir(os.path.join(target_directory, item))]

    # 对于每一个样本的文件夹里面的nii文件进行处理：这里的oneChildFolder是绝对路径
    for oneChildFolder in subdirectories:
        # 使用 os 模块的 rename 函数对nii进行重命名：需要绝对路径
        oneNiiFileList = [file for file in os.listdir(oneChildFolder) if file.endswith('.nii.gz')]
        assert len(oneNiiFileList)==1,f"{oneChildFolder}中，初始的特定名称的nii.gz文件数量必须为1"
        if len(oneNiiFileList)==1:
            input_image = oneNiiFileList[0]
        old_filename = os.path.join(oneChildFolder, input_image)
        temp_new_filename = os.path.join(oneChildFolder, f"{modalityName}_{os.path.basename(oneChildFolder)}.nii.gz")
        os.rename(old_filename, temp_new_filename)

        new_fileName = os.path.join(target_directory, f"{modalityName}_{os.path.basename(oneChildFolder)}.nii.gz")

        # 移动文件并重命名
        shutil.move(temp_new_filename, new_fileName)

        # 删除子文件夹
        if not os.listdir(oneChildFolder):
            shutil.rmtree(oneChildFolder)
    
    # target_directory文件夹里面的文件
    nii_Files = [file for file in os.listdir(target_directory) if os.path.isfile(os.path.join(target_directory, file))]


    # 【四】删除source文件夹里面没有的样本
    delete_time = 0
    ### 遍历source_directory转化，生成的nii文件在子文件夹里面
    for sourceItem in os.listdir(source_directory):
        temp = sourceItem.split("-I")[0]
        final_baseName = f"{modalityName}_{temp}.nii.gz"
        if final_baseName not in nii_Files:
            print("删除")
            delete_time = delete_time+1

    print(f"删除次数:{delete_time}")


    # 【五】对target_directory重新排序并加上前缀
    nii_Files.sort()  # 【需要先排序，以便所有模态的数据能够对齐】
    for idx, file in enumerate(nii_Files, start= rename_start_index):

        old_name = os.path.join(target_directory, file)
        new_name = os.path.join(target_directory,f"{idx:04d}_{file}")

        os.rename(old_name, new_name)

# 测试使用的简单例子
# import dicom2nifti
# sourceChildFolder = "/my_data/dataset/ADNI_1207/2_choose_need_images/sMRI/T1_0_Raw/018_S_0682-20080722-I117870"
# dicom2nifti.convert_directory(sourceChildFolder,output_folder="/my_data/dataset/ADNI_1207/2_choose_need_images/sMRI/Temp")



########################################################################################################【新的使用】
# extract_and_rename_files(
#                 source_directory="/my_data/dataset/ADNI/2_choose_need_images/ADNI_sMRI_and_fMRI/Fun_0_Raw",
#                 target_directory="/my_data/dataset/ADNI/2_choose_need_images/ADNI_sMRI_and_fMRI/Fun_1_NII", rename_start_index=1)

# extract_and_rename_files(
#                 source_directory="/my_data/dataset/ADNI/2_choose_need_images/ADNI_sMRI_and_fMRI/T1_0_Raw",
#                 target_directory="/my_data/dataset/ADNI/2_choose_need_images/ADNI_sMRI_and_fMRI/T1_1_NII", rename_start_index=1)

# extract_and_rename_files(
#                 source_directory="/my_data/dataset/ADNI/2_choose_need_images/ADNI_add_sMRI/T1_0_Raw",
#                 target_directory="/my_data/dataset/ADNI/2_choose_need_images/ADNI_add_sMRI/T1_1_NII", rename_start_index=594)



extract_and_rename_files(
                source_directory="/my_data/dataset/ADNI_1207/2_choose_need_images/sMRI/T1_0_Raw",
                target_directory="/my_data/dataset/ADNI_1207/2_choose_need_images/sMRI/T1_1_NII", rename_start_index=1)



