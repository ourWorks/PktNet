import os

######### 【真的是因为同一个subject同一study_date的原因而导致数量减少为493吗】########

##############将下载的ADNI数据集中的T1_0_Raw寻找出来，并且放进文件夹中：按照subjectId-studyDate的方式命名##############
# 最大的目的是加速下面那一个步骤的转化

# # 保留的文件路径一：：关于1207
base_directory = "/my_data/dataset/ADNI_1207/1_download_data/sMRI"
T1_0_Raw = f"/my_data/dataset/ADNI_1207/2_choose_need_images/sMRI/T1_0_Raw"


if not os.path.exists(T1_0_Raw):
    os.makedirs(T1_0_Raw, exist_ok=True)

image = 0  # 看进入多少次含有dcm文件的imageId文件夹
needSet = set() # needSet的长度就代表，创建了多少个文件夹软链接

# 遍历路径下的子文件夹
for root, dirs, files in os.walk(base_directory):

    if files: # 如果当前文件夹中包含文件

        # image += 1

        # Image里面的第一个文件
        first_file_name_in_leave_folder = files[0]
        print("含有文件的叶结点文件夹路径:", root)
        print("叶结点文件夹中，第一个文件的名称:", first_file_name_in_leave_folder)

        imageId = os.path.basename(root)
        #######################################【sample_name】#############################################
        folder_name = os.path.dirname(root)
        study_date = os.path.basename(folder_name)
        study_year_month_day = study_date[0:10].replace("-","")

        folder_with_objectId_modality = os.path.dirname(folder_name)
        description = os.path.basename(folder_with_objectId_modality)
        objectId = os.path.basename(os.path.dirname(folder_with_objectId_modality))

        sample_name = f"{objectId}-{study_year_month_day}-{imageId}"
        ##################################################################################################

        # 增加的代码：：执行这一句话，能跳过fMRI
        if(description == 'Resting_State_fMRI'):
            continue

        source_directory = root
        # MPRAGE类型的数据也不能在我目前的.mat脚本中进行处理：我目前的脚本只能处理MPRAGE_SENSE2和Resting_State_fMRI的
        # 【MPRAGE_SENSE2】的提取，注意目前的表达式提取到的是带下划线的，也就是MPRAGE_SENSE2_

        # if description=="MPRAGE":
        #     image += 1
        #     symlink_directory = os.path.join(T1_0_Raw,sample_name)
        #     count_deplicated_sampleName = os.path.join(T1_0_Raw,sample_name.split("-I")[0])
        #     try:
        #         if not dirs and files and os.path.exists(symlink_directory):
        #             print(f"软链接{symlink_directory}已存在！")
        #             continue
        #         if count_deplicated_sampleName in needSet: # 【这里千万不要写break】
        #             continue
        #         os.symlink(source_directory, symlink_directory, target_is_directory=True)
        #         print(f"文件夹的软链接已创建: {symlink_directory} -> {source_directory}")
        #         needSet.add(count_deplicated_sampleName)
        #     except OSError as e:
        #         print(f"创建软链接时出错: {e}")
        # else:
        #     break

        image += 1
        symlink_directory = os.path.join(T1_0_Raw,sample_name)
        count_deplicated_sampleName = os.path.join(T1_0_Raw,sample_name.split("-I")[0])
        try:
            if not dirs and files and os.path.exists(symlink_directory):
                print(f"软链接{symlink_directory}已存在！")
                continue
            if count_deplicated_sampleName in needSet: # 【这里千万不要写break】
                continue
            os.symlink(source_directory, symlink_directory, target_is_directory=True)
            print(f"文件夹的软链接已创建: {symlink_directory} -> {source_directory}")
            needSet.add(count_deplicated_sampleName)
        except OSError as e:
            print(f"创建软链接时出错: {e}")


print(f"进入{image}个Image文件夹")
print(f"同一个subject同一study_date只需要一张影像，本次已创建{len(needSet)}个需要的软链接")