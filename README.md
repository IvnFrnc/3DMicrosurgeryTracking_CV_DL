# Tracking the 3D motion of instruments in microsurgery using a marker-based method and the YOLO deep learning method.

### Background

The identification and tracking of tool movements in microsurgery using computer vision can be a potentially useful tool for objectively evaluating surgeons' psychomotor skills, describing a surgical procedure, detecting errors, generating alerts, interacting with and providing information for training assistant robots, among other applications. In this repository, you will find the scripts, datasets, models, and instructions (from scratch) to simultaneously identify and track the 3D movement of two surgical tools during peg transfer microsurgery practice. The images analyzed in this project were recorded using the Mitracks3D exoscope system cameras, consisting of two stereoscopically configured cameras placed under the surgical microscope oriented at a 45-degree perspective towards the surgical target plane, with its center 15 cm away from the camera center. Stereo videos of the peg transfer procedure were recorded. The recording resolution was 1920 x 1080 pixels, captured at a rate of 30 frames per second, with an image offset between the cameras of less than 33 milliseconds in all cases, storing cropped images of 640 x 640 pixels concentric with respect to the center of the original images. 

A dataset was constructed to train YOLO deep learning models by manually labeling 1000 randomly selected images from various recorded trials. These images contained tweezers as the sole label. The tweezers were marked with a 1-millimeter diameter green dot for the right hand and a fuchsia dot for the left hand. These markers help identify and track the tweezers using the color tracking algorithm and help determine if the tweezers are from the left or right hand once identified by the YOLO method.

In a brief description of the findings, it was concluded that the YOLO deep learning tracking system is as good as the color tracking method in terms of precision and accuracy. In terms of processing speed, it was observed that both the color tracking method and the YOLOv8n model are capable of identifying the instruments in real-time using a GPU with CUDA (GTX1660 TI) or superior (results improved using an RTX4060 GPU but were not reported).


# Overview

In this repository, you will find scripts, pre-trained YOLO deep learning models, a database for training YOLO deep learning models for the detection of surgical instruments (tweezers) in microsurgery, including images and labels, configuration files, and training scripts for Google Colab. It also includes execution instructions for processing stereoscopic images recorded with the mitracks3D system and identifying and tracking the 3D movement of two tweezers during microsurgery training on peg transfer using color marker-based tracking methods and YOLO deep learning tracking by Ultralytics.

## Summary

You can use the pre-trained YOLO models included in this repository or train the YOLO models using the provided instructions, scripts, and databases. With the `3D_MicrosurgeryTracking.py` script, you can process stereoscopically recorded images of a procedure using YOLO models and the Mitracks3D method. This script generates a video showing instrument tracking and produces two text files containing the 3D movement records for the color marker-based tracking method and the YOLO method. Each file has six columns: three for the right hand and three for the left hand, with the number of rows depending on the number of frames in the trial. These output files should be formatted using the MATLAB script `FormatData4Analysis.m`, which converts to millimeters, translates, and rotates the point cloud to align with the tracking system's center. This script generates a text file that can be used to analyze surgical maneuver movements.

For analyzing surgical maneuvers, a MATLAB script called `Analysis_MAPs.mlx` is included. For the reader's convenience, this script automatically loads and processes the movement records of 35 participants, contained in the `MATLAB/MAPs_Statistics_Comparison/A_OK` folder. These data are used to calculate six motion analysis parameters (MAPs) in 3D and 2D, such as time, path length, bimanual dexterity, distance between instruments, number of submovements, and speed. Subsequently, a statistical analysis is conducted to determine if there were statistically significant differences between the participants' scores, and finally, the similarities between the two 3D and 2D recording modalities are calculated and graphically displayed using cosine similarity and normalized score graphs.

As an example, a script within the `MATLAB/Plot3D_VisualComparison` folder is provided, allowing for a visual comparison of the 3D components of the movement records using Mitracks3D and YOLOv8.

To demonstrate the identification and tracking of instruments in stereoscopic images for 3D maneuver recording, the video `TrackingMitracks3DvsYOLOv8.mp4` is provided. This video shows the instrument tracking using the color marker-based method (top section with circle detection) and the YOLO deep learning method (bottom section with rectangles). On the right side of the video, the 3D component traces generated by both methods for both hands are displayed.

## Requirements
Python3, MATLAB 2021, OpenCV 4, CUDA 12, Ultralytics YOLOv8.  

## Files Listed in Order of Use

The procedure should be followed in the order indicated in this document.

### 1. Files for Training YOLO Models for Tweezer Identification in Microsurgery

(Note: You can use the pre-trained models included in the `Tracking3D` folder and skip to step 2).

The `DatasetMitracksYOLO` folder contains:
1. `Tutorial_TrainingYOLOinGoogleColab.txt` - Tutorial with instructions for training YOLO models for surgical instrument identification.
2. `TrainYOLOinCOLAB_4Mitracks.ipynb` - Google Colab script, import it to Colab according to the instructions in the previous file.
3. `config_train_COLAB_2Tweezers.yaml` - Configuration file for training.
4. `datasets` directory - Training dataset containing 1000 images and their respective labels.

**Inputs:** Dataset, configuration file  
**Outputs:** Trained YOLOv8X model(s) for tweezer detection in microsurgery

### 2. The `Tracking3D` Folder Contains the `3D_MicrosurgeryTracking.py` Script

This script records the 3D movement of two tweezers during microsurgery practice from stereoscopic images using the color marker-based tracking algorithm and the YOLO deep learning method simultaneously. 

To test the script's functionality, ensure your PC supports CUDA for processing algorithms on the GPU. Please install Ultralytics to use YOLO, e.g., `pip install ultralytics`. Ensure you have OpenCV (version 4.8 or higher) for Python installed, or install it.

Once the prerequisites are installed, copy the `Scripts` folder with all its contents. Run the `3D_MicrosurgeryTracking.py` script.

The `Tracking3D` folder includes:
1. The `3D_MicrosurgeryTracking.py` script for 3D tracking of instruments in microsurgery.
2. `example_trial1` folder, including stereoscopic images of a complete procedure.
3. Pre-trained YOLO models `best_Mitracks3D_YOLOv8n.pt` (6.1 MB) and `best_Mitracks3D_YOLOv8m.pt` (50.815 MB).

The `3D_MicrosurgeryTracking.py` script will generate two `.txt` files, `data_3D_mitracks.txt` and `data_3D_YOLO8.txt`, as well as a video `Instruments_tracking.mp4`, and save them within the `example_trial1` folder. These files contain the tracking data of the surgical tools using both the color marker-based tracking method and the YOLO deep learning method. These files have six columns: three for [x, y, z] of the right hand and three for [x, y, z] of the left hand. The number of rows corresponds to the number of frames in the procedure recording.

**Script:** `3D_MicrosurgeryTracking.py`  
**Inputs:** `example_trial1` (dataset), YOLO model `best_Mitracks3D_YOLOv8X.pt`  
**Outputs:** `data_3D_mitracks.txt`, `data_3D_YOLO8.txt`

4. The `OutputsExample` folder contains examples of outputs from the last described script.

5. `data_3D_mitracks.txt` - An example of a 3D record of a peg transfer procedure. This is an example output of the previously mentioned script, and these data should be reformatted with the following file.
6. `FormatData4Analysis.m` is a MATLAB script that converts to millimeters, rotates, and translates the 3D record point cloud to align it with the real-world coordinates and make it compatible with the recording system.
7. `output_Data3D_2b_analized.txt` is an example output of the formatting script, prepared for analysis using the scripts mentioned below.

**Script:** `FormatData4Analysis.m`  
**Inputs:** `data_3D_mitracks.txt` or `data_3D_YOLO8.txt`  
**Outputs:** `output_Data3D_2b_analized.txt`

### 3. Analysis and Visualization of Movement Tracking Data in MATLAB

The `MATLAB` folder contains the following:

#### 4.1 The `MAPs_Statistics_Comparison` Folder Contains

1. The `A_OK` folder, which includes the database of formatted 3D records ready for analysis. It contains a total of 105 files corresponding to the records of 35 participants using the color tracking method, YOLOv8n in the `Av8n` folder, and YOLOv8m in the `Av8m` folder.
2. `Analysis_MAPs.mxl` is a cell-execution MATLAB script. This script loads the database of 3D records and calculates the six MAPs in 3D and 2D; with the scores of these MAPs, it calculates the Mann-Whitney U test to find statistically significant differences between the two recording modalities. It compares the cosine similarity between the modalities and normalizes the scores for graphical comparison.

**Script:** `Analysis_MAPs.mxl`  
**Inputs:** `A_OK` (3D record database)  
**Outputs:** Multiple outputs and graphs (see result tables and figures in the script)

#### 4.2 The `Plot3D_VisualComparison` Folder Contains

1. `Plot_3D_VisualComparation.mxl` - MATLAB script to graph and visually compare the 3D components (x, y, z) of the records using the color tracking method and the YOLO method.
2. Examples of a peg transfer trial, these files are pre-formatted (no preprocessing required): `ExampleData_Mitracks3D.mat`, `ExampleData_YOLOv8n.mat`, `ExampleData_YOLOv8m.mat`.

**Script:** `Plot_3D_VisualComparation.mxl`  
**Inputs:** `ExampleData_XX.mat`  
**Outputs:** Multiple outputs, see graphs within the script execution notebook.

### 4. Video Demonstration

To illustrate the transfer of objects, the video `Example_First_6_transfers.mp4` was added
In the following link you can find a video demonstrating the detection and tracking of 3D movement using both methods: color marker-based tracking and YOLO deep learning tracking: https://youtu.be/TCq9_qYcfXo
