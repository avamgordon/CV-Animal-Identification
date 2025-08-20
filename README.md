# CV-Animal-Identification
Programming final project completed for CSE 5524 (Computer Vision) using Python for multi-species individual animal reidentification. 

The project report for a full description of this project is available for read in this repository, titled "Final Project Report". Project code is also available. To generate results, decompress the .zip file, open the folder in your preferred platform, and run the test.py file which will generate a submission .csv file. 
Then, visit https://www.kaggle.com/competitions/animal-clef-2025 and press “Submit Prediction” in the top right-hand corner and upload the submission .csv file.

Python packages the need installed are pytorch, numpy, cv2, sklearn, pandas, and pillow.
*** Make sure the constant CALCLUATE_THRESHOLD is set to False, and the constant PREDICT_TEST_DATA is set to True before running test.py ***

*** Changing the line "datapath = 'datasets/animal-clef-2025/metadata.csv'" to "datapath = 'datasets/animal-clef-2025/metadata_small.csv' will run a short subsample of all the data ***
