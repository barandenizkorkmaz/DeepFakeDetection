# CMPE 492: DeepFake Detection - v3 - Download Instructions

This file describes the steps for downloading the `FaceForensics++` dataset. In this version, we will be able to download our dataset into a separate directory so that it can reside in a directory that is independent from the directory where the code for our model is located.

1. Create the directory where you want the data to be located. We recommend such a directory tree structure:

```
|-- ..
    |-- CMPE492_DeepFakeDetection_FaceForensics
        |-- LQ
        |-- HQ
        |-- RAW
```

For convenience, we will call `DATA_DIR` for the absolute path of directory which the downloaded dataset is located for the rest of the document. For the file structure above, if you want to work with HQ videos, then we call `DATA_DIR` for the following path:

`DATA_DIR = ../CMPE492_DeepFakeDetection_FaceForensics/HQ`

`IMPORTANT NOTE:` The download script provided by Technical University of Munich works only if the names of folders are formatted in FAT32. For simplicity, please use only alphanumerical characters and underline.

2. Inside `DATA_DIR`, download the script called `faceforensics_download_v4.py` provided by Technical University of Munich for FaceForensics++ dataset. You could find the download script in your mail inbox.

   1. After the download has finished, open the script and comment `line 143`:

      ```python
      # _ = input('')
      ```

3. Clone our GitHub repository into any directory you want by entering the following command in terminal and navigate into the repository folder.

   ```bash
   git clone https://github.com/barandenizkorkmaz/DeepfakeDetection-v3-1
   cd DeepfakeDetection-v3-1
   ```

4. Copy the download script from our repository folder into `DATA_DIR` by entering the following command.

   ```bash
   cp download_script.py [DATA_DIR]
   ```

   At this stage, our suggested directory tree structure should look like as follows:

   ```
   |-- ..
       |-- CMPE492_DeepFakeDetection_FaceForensics
           |-- LQ
           |-- HQ
           	|-- download_script.py
           	|-- faceforensics_download_v4.py
           |-- RAW
   ```

   where  `DATA_DIR = ../CMPE492_DeepFakeDetection_FaceForensics/HQ` .



5. Navigate into `DATA_DIR` and run the download script.

   ```bash
   cd [DATA_DIR]
   python3 download_script.py -q [lq|hq|raw]
   ```

   Arguments:

   1. -q: Video Quality
      1. lq: Low-Quality (c40)
      2. hq: High-Quality (c23)
      3. raw: Raw

   Since we are going to work with high-quality videos, you should enter the following command:

   ```bash
   python3 download_script.py -q hq
   ```

   

6. After the download has been successfully completed, the directory tree structure should be as follows:

```
|-- ..
    |-- CMPE492_DeepFakeDetection_FaceForensics
        |-- LQ
        |-- HQ
        	|-- manipulated_sequences
        	|-- original_sequences
        	|-- download_script.py
        	|-- faceforensics_download_v4.py
        |-- RAW
```

7. Navigate back into our repository folder and then copy the scripts and files required for creating dataset which will be used for training model into `DATA_DIR` .

```bash
cp select_video_subset.py [DATA_DIR]
cp extract_frames.py [DATA_DIR]
cp select_frames.py [DATA_DIR]
cp requirements.txt [DATA_DIR]
```

Recall that `DATA_DIR = ../CMPE492_DeepFakeDetection_FaceForensics/HQ` .

8. Run the script that will produce a subset of FaceForensics++ dataset in a way that the number of real and fake videos and the distribution of fake creation techniques are distributed evenly. This script will create a subset of FaceForensics++ dataset into the folder called `FaceForensicsDFD`.

```bash
python3 select_video_subset.py
```

9. Copy the scripts that are required for creating our own dataset into `FaceForensicsDFD` folder.

```bash
cp extract_frames.py FaceForensicsDFD
cp select_frames.py FaceForensicsDFD
cp requirements.txt FaceForensicsDFD
```

10. Navigate into `FaceForensicsDFD`.

```bash
cd FaceForensicsDFD
```

11. Create a virtual environment called `env` (you can choose any name) inside `FaceForensicsDFD`.

```bash
python3 -m venv env
```

 1. Activate the virtual environment called `env`

    ```bash
    source env/bin/activate
    ```

    

 2. Make sure that your pip is referenced to pip3. Otherwise, please do not proceed ahead and contact us.

    ```
    pip -V
    ```

    

 3. Upgrade the pip of your virtual environment.

    ```
    pip install --upgrade pip
    ```

    

 4. Install the required packages into the virtual environment.

    ```
    pip3 install -r requirements.txt
    ```


12. At this stage, the directory tree structure should be as the following:

```
|-- ..
    |-- CMPE492_DeepFakeDetection_FaceForensics
        |-- LQ
        |-- HQ
        	|-- FaceForensicsDFD
        		|-- manipulated_sequences
        		|-- original_sequences
        		|-- env
        		|-- requirements.txt
        		|-- extract_frames.py
        		|-- select_frames.py
        	|-- manipulated_sequences
        	|-- original_sequences
        	|-- download_script.py
        	|-- faceforensics_download_v4.py
        	|-- select_video_subset.py
        	...
        |-- RAW
```



13. Extract the frames within video sequences.

```bash
python3 extract_frames.py
```



14. Generate the dataset named `CMPE492DeepfakedetectionData` from the extracted frames automatically. Beware that this step might last long, since the faces within frames are detected, aligned, and cropped accordingly.

```
python3 select_frames.py
```



15. At this stage, the directory tree structure should be as the following:

```
|-- ..
    |-- CMPE492_DeepFakeDetection_FaceForensics
        |-- LQ
        |-- HQ
        	|-- FaceForensicsDFD
        		|-- manipulated_sequences
        		|-- original_sequences
        		|-- CMPE492DeepfakedetectionData
        			|-- Training
        				|-- Training
        					|-- Fake
        					|-- Real
        				|-- Validation
        					|-- Fake
        					|-- Real
        			|-- Test
        				|-- Fake
        				|-- Real
        		|-- env
        		|-- requirements.txt
        		|-- extract_frames.py
        		|-- select_frames.py
        	|-- manipulated_sequences
        	|-- original_sequences
        	|-- download_script.py
        	|-- faceforensics_download_v4.py
        	|-- select_video_subset.py
        	...
        |-- RAW
```

Please note that, while training our model, you will provide the path of `CMPE492DeepfakedetectionData` as the argument. We will later provide a readme file for training instructions.