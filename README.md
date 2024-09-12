# PktNet

## 1. dataset

The ADNI dataset is used in our work.
Due to regulations, we cannot directly provide the ADNI dataset. Anyone must apply for access through the ADNI website to obtain the required data.
All used imageIds are listed in "dataset/imageIds.txt". After copying one of the IDs into the advanced search, you can search and download the image on the website <https://adni.loni.usc.edu>.

## 2. preprocessing

The editor can access the processed files through the following link. 
**The password is a combination of the first letters of the first four words in the article title, all in lowercase.**

https://pan.baidu.com/s/1VStSEheMZ1k1Ql0siyNx8A

First, download oral SMRIs data and extract them in folder "1_downloadSearchCSVByimageIds_and_diagnosisCSV".
Second, combine the CSV files to generate the label in folder "2_getLabelFromLastTwoCsvs".
Third, extra the files in folder "3_download_and_Preprocess_SMRIS_samples", and preprocess the data with the scripts in folder "preprocess_SMRI".

```cmd
# to get the needed structure dataset.
python preprocess_SMRI/A1_linkFromOral.py 
# to these 2D image files into 3D files containing the scalp and skull
python preprocess_SMRI/B1_toNii.py 
# finish the remaining steps
python preprocess_SMRI/B2_genSMRIFile.py
```

1). **Refine 3D Images**:

- Remove the skull and most blank areas from the 3D images to obtain a refined image.
- Generate an abstract representation of the refined image.

2). **Spatial Registration**:

- Perform spatial registration on the abstract representation to obtain a registered image.
- Remove the surrounding white space from the registered image.

3). **Overlay AAL3 Template**:

- Overlay the AAL3 template onto the processed image to visualize overlapping areas.

4). **Resize and Normalize**:

- Resize each sMRI image to 96×96×96.
- Normalize voxel intensities to the range (0, 128).

Now we can copy files to folder "4_processed_files", the files is provided in the link.
Finally, we can arrange the folder "4_processed_files" and "2_getLabelFromLastTwoCsvs/result_labels.csv" to generate the json files in "5_link_dataset", and the json files are copied to folder "data" in this project.

## 3. preprocess example

The preprocess example can be seen in folder "preprocess_example"，and it will be upload soon.

## 4. training

You have to train with several params in the file "train.py". For example, you can train with the command below.

```cmd
python train.py --model PktNet --num_classes 2 --json_path data/pureSMRI_1207_threeStage/twoClass_5Fold/CNvsAD/fiveFold_3.json
```

## 5. testing

The result of our work is shown in folder "experiment_result". It can be downloaded from the link: https://pan.baidu.com/s/1DAXxfwrEKUPu1NvGPc2jcw?pwd=1688
You can modify the default pth path in file "testBySavedPth.py" and other params to test the model, and run the command below.

```cmd
python test.py
```

## 6. explain

The explain module will be uploaded soon. **We will upload a video to show our work end-to-end after the paper is accepted.**
