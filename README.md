# PktNet

## 1. dataset

The ADNI dataset is used in our work.
Due to regulations, we cannot directly provide the ADNI dataset. Anyone must apply for access through the ADNI website to obtain the required data.
All used imageIds are listed in "dataset/imageIds.txt" and the corresponding subjectIds are listed in the file "subjects_of_imageId.txt". After copying one of the IDs into the advanced search, you can search and download the image on the website <https://adni.loni.usc.edu>.

**If you need to download all of the images listed in the file at once, connect all of the imageIds with commas and paste them into the corresponding search box in advanced search on the ADNI website.**

## 2. preprocessing

The raw data is preprocessed using the following steps:

1). **Convert**:

- Convert these 2D image files into 3D files containing the scalp and skull.

2). **Refine 3D Images**:

- Remove the skull and most blank areas from the 3D images to obtain a refined image.
- Generate an abstract representation of the refined image.

3). **Spatial Registration**:

- Perform spatial registration on the abstract representation to obtain a registered image.
- Remove the surrounding white space from the registered image.

4). **Overlay AAL3 Template**:

- Overlay the AAL3 template onto the processed image to visualize overlapping areas.

5). **Resize and Normalize**:

- Resize each sMRI image to 96×96×96.
- Normalize voxel intensities to the range (0, 128).

## 3. training

If the editor requests or the paper is confirmed to be accepted, we will publish more information about it.

## 4. testing

If the editor requests or the paper is confirmed to be accepted, we will publish more information about it.

## 5. result

The simple result of our work is now shown in folder "pureSMRI_1431_threeStage_results". You can download the results with weights and overwrite the simple result to add the trained weight. It can be downloaded from the link below.

- <https://drive.google.com/drive/folders/1INIrA762tvRE2fYH1MKb9zYl7uwDxdBn?usp=sharing>
