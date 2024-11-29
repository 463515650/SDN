# SDN
This is the source code of SDN.

Our paper, "Anomaly Diagnosis with Siamese Discrepancy Networks in Distributed Cloud Databases", is submitted to ICDE2025.

## Directory description
  * DBSherlock: The baseline, DBSherlock of our implementation.
  * ISQUAD: The baseline, ISQUAD of our implementation.
  * preprocess: Code for data preprocessing and data construction.
  * model.py: Our implementation of SDN.
  * eval.py: Code used to train, test and interpret with SDN.

## Linux build
SDN is implemented in Python 3.10 with PyTorch 2.0.1. All experiments are run on a Linux (Ubuntu 18.04LTS) machine with a NVIDIA Tesla V100 GPU card. 
### Prerequisites
 * Python 3.10
 * pip 22.2.2

### Build and run steps
 * Download this repository and change to the root folder.

 * Installing dependencies

   `$ pip install -r requirements`

 * Preprocess and constructure the training data

   `$ python preprocess/sample_generator.py -k 5 -test_ratio 0.5 -sample_strategy clustered`

 * Train SDN and evaluate

   `$ python eval.py -mtcn_c 4 -p 2 -q 2 -latent_size 32`