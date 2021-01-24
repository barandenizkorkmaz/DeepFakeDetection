# CMPE 492: DeepFake Detection - v3

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

   

6. After the download has been successfully completed, the file structure should be as follows:

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

