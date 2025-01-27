# CAAT-EHR: Cross-Attentional Autoregressive Transformer for Multimodal Electronic Health Record Embeddings
CAAT-EHR, a novel architecture designed to generate robust, task-agnostic longitudinal embeddings from raw EHR data. CAAT-EHR leverages self- and cross-attention mechanisms in its encoder to integrate temporal and contextual relationships across multiple modalities, transforming the data into enriched embeddings that capture complex dependencies. An autoregressive decoder complements the encoder by predicting future time points data during pre-training, ensuring that the resulting embeddings maintain temporal consistency and alignment. CAAT-EHR eliminates the need for manual feature engineering and enables seamless transferability across diverse downstream tasks.
After pre-training CAAT-EHR, only the trained CAAT-EHR encoder is retained. This encoder is then used to generate a new representation: the task-agnostic longitudinal embedding for the external data.
![CAAT-EHR](https://github.com/bozdaglab/CAAT-EHR/blob/main/Images/CAAT-EHR.png?raw=true)
## Dataset and input format
CAAT-EHR was pre-trained independently on two datasets. The first dataset, the Alzheimer’s Disease Neuroimaging Initiative (ADNI), includes two longitudinal modalities: cognitive measurements and MRI data. The second dataset, the Medical Information Mart for Intensive Care (MIMIC-III), includes two longitudinal modalities: continuous data and categorical data.
- To access the ADNI dataset, you need to request access through https://adni.loni.usc.edu/
- To access the MIMIC-III, you need to request access through https://mimic.mit.edu/docs/gettingstarted/

For pre-training, the dataset was partitioned into input features and prediction targets. In time series EHR data with 𝑇 time points (visits), the data from the first time point (visit) to 𝑇 – 2 were used as input features, while data from 𝑇 − 2 to the last time point served as prediction targets. The CAAT-EHR model was trained to predict the target data based on the input features.
To conduct pre-training for CAAT-EHR, three inputs representing the same samples are required:
1. The first data modality, structured as (number of samples, maximum number of time points, number of features for the first data modality).
2. The second data modality, structured as (number of samples, maximum number of time points, number of features for the second data modality).
3. The target data, structured as (number of samples, 2, number of features for the first data modality+ number of features for the second data modality).

For the first and second data modalities, any samples with fewer time points than the maximum number of time points were padded at the end with a value of -50.
## Compatibility
All codes are compatible with Tensorflow version 2.14.0, Keras version 2.14.0 and Python 3.11.5.
## How to run CAAT-EHR
To run CAAT-EHR, ensure the following files are in the same directory:
1. CAAT-EHR.ipynb: The implementation of the proposed model, and its located in 'CAAT-EHR model' folder.
2. modal1.pkl: Represents the first data modality, and its located in 'Sample of Pre-training data' folder.
3. modal2.pkl: Represents the second data modality, and its located in 'Sample of Pre-training data' folder.
4. target.pkl: Represents the target data to be predicted, and its located in 'Sample of Pre-training data' folder.

Once all files are in the same directory, open and execute CAAT-EHR.ipynb using Jupyter Notebook. During execution, CAAT-EHR will be pre-trained, and the encoder portion of the model will be saved as a Keras model. This saved encoder can later be used to generate longitudinal embeddings for external EHR data.
***Note:*** The external EHR data should have the same number of data modalities as those used during the pre-training phase.
## How to generate embeddings for an external data
To generate embeddings for external data, ensure that the external EHR data has the same number of data modalities as those used during the pre-training phase and follows the format described in the dataset section. To proceed, make sure the following files are in the same directory:
1. Generate_embeddings.ipynb: The script for generating embeddings, and its located in 'Generating embeddings' folder.
2. transformer_encoder.keras: The pre-trained encoder model, and its located in 'Generating embeddings' folder.
3. external_modal1.pkl: Represents the first data modality, and its located in 'Samples of external data' folder.
4. external_modal2.pkl: Represents the second data modality, and its located in 'Samples of external data' folder.

Once all files are in place, open and execute Generate_embeddings.ipynb using Jupyter Notebook. Upon completion, the generated embeddings will be structured as (number of samples, maximum number of time points, embedding size) and saved as a pickle file.
These embeddings are now longitudinal and task-agnostic and can be fed into any sequential model such as an RNN or Transformer or even aggregated to be used with models like MLP, Random Forest (RF), or Support Vector Machine (SVM) for downstream tasks.

***Note:*** The pre-training and external data provided in this repository were randomly created .
