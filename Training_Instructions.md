# CMPE 492: DeepFake Detection - v3 - Training Instructions

This file describes the steps for training our model.

1. Clone our GitHub repository into any directory you want by entering the following command in terminal and navigate into the repository folder. We will call the directory of our repository folder `REPO_DIR`.

   ```bash
   git clone https://github.com/barandenizkorkmaz/DeepfakeDetection-v3-1
   cd DeepfakeDetection-v3-1
   ```

2. We will call `DATA_DIR` as the directory where the dataset is located. Its content should look as follows:

   ```
   |-- ..
   	|-- DATA_DIR
   		|-- deepfake_aligned
   		|-- face2face_aligned
   		|-- faceswap_aligned
   		|-- ff_real_aligned
   		|-- ...
   ```

3. In `REPO_DIR`, obtain the re-arranged form of the content of `DATA_DIR` (without changing the original files) in another folder. 

   ```bash
   python3 directory_manipulation.py -src [DATA_DIR] -dest [TARGET_DIR]
   ```

   `TARGET_DIR` should be a user-defined directory where you want the dataset to be located. You will use `TARGET_DIR` while entering the path of dataset for training.

4. Create a virtual environment called `env` and activate the virtual environment you have created. (You can give any name for virtual environment.)

   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

5. Upgrade the pip of virtual environment and install the required packages into virtual environment.

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   

6. Add the path of current working directory into `PYTHONPATH` environment variable.

   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   ```

7. Run the program.

   ```bash
   python3 main.py -d [TARGET_DIR] -e [NUM_EPOCHS] -b [BATCH_SIZE]
   ```

   

   1. TARGET_DIR: The directory of dataset that you have previously created in step (3).
   2. NUM_EPOCHS: Number of epochs. (Default = 20)
   3. BATCH_SIZE: Batch size. (Default = 32)

The program will check whether the current working directory (`REPO_DIR`) contains a model named `DeepfakeDetection_Model`.

* If there is a pretrained model, the program will simply import the pretrained model and make predictions on the test data.
* Else, the program will train a new model from scratch and make predictions on test data.