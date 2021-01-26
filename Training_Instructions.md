# CMPE 492: DeepFake Detection - v3 - Training Instructions

This file describes the steps for training our model. These instructions must be followed after the instructions in `Download_Instructions.md` have been successfully completed.

1. Navigate into `DATA_DIR/FaceForensicsDFD` and activate the virtual environment that you have previously created and installed the required packages.

   ```
   cd [DATA_DIR]/FaceForensicsDFD
   source env/bin/activate
   ```

   

2. Navigate into repository folder.

3. Add the path of current working directory into `PYTHONPATH` environment variable.

   ```
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   ```

   

4. Run the program.

   ```
   python3 main.py -d [DATASET_DIR] -e [NUM_EPOCHS] -b [BATCH_SIZE]
   ```

   

   1. DATASET_DIR: The directory of dataset that you have previously created. It is in the following format `../CMPE492_DeepFakeDetection_FaceForensics/HQ/FaceForensicsDFD/CMPE492DeepfakedetectionData` if you have followed the suggested directory structure.
   2. NUM_EPOCHS: Number of epochs. (Default = 20)
   3. BATCH_SIZE: Batch size. (Default = 32)

The program will check whether the current working directory contains a model named `DeepfakeDetection_Model`.

* If there is a pretrained model, the program will simply import the pretrained model and make predictions on the test data.
* Else, the program will train a new model from scratch and make predictions on test data.