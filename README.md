[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15095937.svg)](https://doi.org/10.5281/zenodo.15095937)

# FastText2Doc2Vec-Doc-relevance-training

This repository focuses on an approach exploring and assessing literature-based doc-2-doc recommendations using the fastText algorithm with its application to the RELISH dataset.  The dataset used is the RELISH Corpus, an expert-curated collection of biomedical literature consisting of pairwise document assessments. The workflow involves training the fastText models on a specified training set and then evaluating the document-to-document recommendations on a separate test set. Additionally, we employ Optuna for optimizing the hyperparameters for the trained fastText models.

## 📚🔍 Table of Contents

- [FastText2Doc2Vec-Doc-relevance-training](#fasttext2doc2vec-doc-relevance-training)
  - [📚🔍 Table of Contents](#-table-of-contents)
  - [📝About](#about)
  - [📂Input Data](#input-data)
  - [🛠️Pipeline](#️pipeline)
    - [🧠⚙️Train and Optimize fastText models](#️train-and-optimize-fasttext-models)
        - [Parameters](#parameters)
  - [💾⚙️Utilizing a Pre-trained Model](#️utilizing-a-pre-trained-model)
  - [📐🔄Calculate Cosine Similarity](#calculate-cosine-similarity)
  - [📈📋 Evaluation](#-evaluation)
    - [🎯Precision@N](#precisionn)
    - [📊 nDCG@N](#-ndcgn)
  - [🧑‍💻🧩 Code Implementation](#-code-implementation)
  - [🚀Getting Started](#getting-started)
    - [Step 1: Clone the Repository](#step-1-clone-the-repository)
          - [Using HTTP:](#using-http)
          - [Using SSH:](#using-ssh)
    - [Step 2: Create a virtual environment and install dependencies](#step-2-create-a-virtual-environment-and-install-dependencies)
    - [Step 3: Dataset](#step-3-dataset)
    - [Step 4: Optimization Pipeline](#step-4-optimization-pipeline)

## 📝About

Our approach employs [FastText](https://fasttext.cc/docs/en/support.htm) to generate word embeddings and subsequently employs a centroid aggregration technique to produce document-level embeddings. This process involves calculating the centroids of word embeddings found in the titles and abstracts of each document. The approach is applied to evaluate literature-based document-to-document recommendations using the RELISH dataset.

## 📂Input Data
The input data for this method includes preprocessed tokens derived from the RELISH documents, a specialized database curated by experts for benchmarking document similarity in biomedical literature. The RELISH dataset comprises a JSON file containing PubMed IDs (PMIDs) along with document-to-document relevance assessments categorized as "relevant," "partial," or "irrelevant." Titles and abstracts of the associated articles were retrieved and stored in a TSV file. 

The title and abstract text are preprocessed, and the resulting tokens are stored in the RELISH.npy file, which includes arrays of PMIDs, document titles, and abstracts. Within this preprocessing pipeline, both the title and abstract texts undergo several stages of refinement: stop words and structural words are eliminated, the text is converted to lowercase, and finally, tokenization is employed, resulting in arrays of individual words and is detailed in the [relish-preprocessing repository](https://github.com/zbmed-semtec/relish-preprocessing). The resulting preprocessed tokens are divided into training, validation and test sets based on specific criteria detailed [here](https://github.com/zbmed-semtec/relish-preprocessing?tab=readme-ov-file#splitting-the-data). These splits are then saved as two separate .npy files.

Additionally, the ground truth relevance assessments are used to evaluate the accuracy of the doc-2-doc recommendations, ensuring that the method's results align with expert judgments. For this, we make use of the validation ground truth TSV file for hyperparameter optimization and the test ground truth TSV file for the final evaluation.

## 🛠️Pipeline

The following section outlines the process of generating document-level embeddings out of word-level embeddings for each PMID of the RELISH corpus through hyperparameter optimization, computing the cosine similarity scores and evaluating the given similarity results with the relevance matrix.

### 🧠⚙️Train and Optimize fastText models
We create and train fastText models with customizable hyperparameters to comprehend the connections between documents and words in a high-dimensional vector space. We aim to optimize these hyperparameters to establish the most effective relationship between cosine similarity and document relevance.

To accomplish this we begin by splitting the dataset into a training set and a testing set. The training set is then used to train the fastText model, where we explore various hyperparameters to optimize its performance. This optimization process is crucial for enhancing the model's ability to capture meaningful relationships between cosine similarity and document relevance. For each set of hyperparameters, a fastText model is trained on the training split.

Following this, we evaluate the model's performance on the testing set using Precision@5 as our evaluation metric.

##### Parameters

+ **sg:** {1,0} Refers to the training algorithm. If sg=1, skip-gram is used otherwise, continuous bag of words is used.
+ **vector_size:** It represents the dimensions of the generated embeddings, with options of 200, 300 and 400 in our case.
+ **window:** Represents the maximum distance between the current and predicted word, with values fof 5,6 and 7 in our case.
+ **epochs:** Refers to the number of iterations over the training dataseta and is set to vary from 5 to 15 in this context.
+ **min_count:** It is the minimum number of appearances a word must have to not be ignored by the algorithm and is configured at 1, 2 and 3 in our case.

## 💾⚙️Utilizing a Pre-trained Model
Additionaly, we leverage the pre-trained 'cc.en.300.bin.gz' fastText model trained on Common Crawl and Wikipedia. Using this model, we generate word level embeddings for the test dataset, calculate centroid of word embeddings to generate document level embeddings, compute cosine similarity between the embeddings and evaluate them using precision@N and nDCG@N metrics.

## 📐🔄Calculate Cosine Similarity

Following hyperparameter optimization where the best model gets saved, embeddings are generated for the test dataset using this trained model. Subsequently, cosine similarity is calculated for the test dataset embeddings, providing a measure of similarity between pairs of documents based on their learned representations. This enables the generation of a 4-column matrix [ PMID1 | PMID2 | Relevance | Cosine similarity ] containing cosine similarity scores for existing pairs of PMIDs within our corpus. For a more detailed explanation of the process, please refer to this [documentation](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/Cosine_Similarity).

## 📈📋 Evaluation

The effectiveness of the embeddings in capturing document-to-document similarity is assessed using two metrics: Precision@N and nDCG@N.

### 🎯Precision@N

Precision@N measures the precision of retrieved documents at various cutoff points (N).We generate a Precision@N matrix for existing pairs of documents within the RELISH corpus, based on the original RELISH JSON file. The [code](code/precision.py) determines the number of true positives within the top N pairs and computes Precision@N scores. The result is a Precision@N matrix with values at different cutoff points, including average scores. For detailed insights into the algorithm, please refer to this [documentation](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/Precision%40N_existing_pairs).

### 📊 nDCG@N

Another metric used is the nDCG@N (normalized Discounted Cumulative Gain). This ranking metric assesses document retrieval quality by considering both relevance and document ranking. It operates by using a TSV file containing relevance and cosine similarity scores, involving the computation of DCG@N and iDCG@N scores. The result is an nDCG@N matrix for various cutoff values (N) and each PMID in the corpus, with detailed information available in the [documentation](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/Evaluation).


## 🧑‍💻🧩 Code Implementation

+ The [`main.py`](code/main.py) serves as a comprehensive wrapper function, supporting the model generation, training, embedding generation, cosine similarity matrix calculation, precision calculation and gain calculation in one pipeline and the final evaluation for the test dataset. Individual functions for each task are provided in the other scripts.

+ [`optunaTuningUnix.py`](code/optunaTuningUnix.py) / [`optunaTuningWindows.py`](code/optunaTuningWindows.py) : The code utilizes Optuna for hyperparameter optimization of fastText model. It suggests hyperparameters for fastText, trains models, evaluates precision@5, and selects the best trial. The optimization process iterates over several trials, updating progress with a progress bar. The scripts are designed to run the pipeline on either Unix or Windows systems.

+ [`train.py`](code/train.py): This script trains a fastText model using specified hyperparameters, saves the model if specified, generates embeddings for test data, computes cosine similarity scores, and saves them to a file. It logs progress to a file specified by log_file.

+ [`pretrained.py`](code/pretrained.py) This script uses the Pre-trained 'cc.en.300.bin.gz' model, generates embeddings and computes cosine similarity directly for the test dataset.

+ [`utilities.py`](code/utilities.py): This script includes functions for parsing and reading input tokens, creation and training of fastText models, generation of embeddings, centroid aggregation of word embeddings to generate document embeddings, calculation of cosine similarity, generation of similarity matrix.

+ [`precision.py`](code/precision.py): This script reads a TSV file containing cosine similarity pairs, calculates precision scores at various values of n for each PMID, and writes the results along with average precision scores to a new TSV file.

+ [`calculate_gain.py`](code/calculate_gain.py): This script calculates normalized discounted cumulative gain (nDCG) scores for relevance assessment based on cosine similarity values, sorts data accordingly, and writes results including average nDCG scores to a TSV file. It utilizes the cosine similarity matrix provided and performs operations per PMID.

## 🚀Getting Started

To get started with this project, follow these steps:

### Step 1: Clone the Repository
First, clone the repository to your local machine using the following command:

###### Using HTTP:

`git clone https://github.com/zbmed-semtec/fasttext2doc2vec-doc-relevance-training.git`

###### Using SSH:
Ensure you have set up SSH keys in your GitHub account.

`git clone git@github.com:zbmed-semtec/fasttext2doc2vec-doc-relevance-training.git`


### Step 2: Create a virtual environment and install dependencies

To create a virtual environment within your repository, run the following command:

```
python3 -m venv .venv 
source .venv/bin/activate   # On Windows, use '.venv\Scripts\activate' 
```

To confirm if the virtual environment is activated and check the location of yourPython interpreter, run the following command:

```
which python    # On Windows command prompt, use 'where python'
                # On Windows PowerShell, use 'Get-Command python'
```
The code is stable with python 3.9 and higher. The required python packages are listed in the requirements.txt file. To install the required packages, run the following command:

```
pip install -r requirements.txt
```

To deactivate the virtual environment after running the project, run the following command:

```
deactivate
```

### Step 3: Dataset


- Use the [Download_Dataset.sh](./Download_Dataset.sh) script to download the Split Dataset by running the following commands:

```
chmod +777 Download_Dataset.sh
./Download_Dataset.sh
```
This script makes sure that the necessary folders are created and the files are downloaded in the corresponding folders.

**OR**


- You could also download the dataset from this link: [Split_Dataset](https://drive.google.com/drive/folders/1Bq_U5207utn7tvSt_HLVdOdYR5QW7MMN). Please make sure to keep the data in the below specified format.

```
📦 /fasttext2doc2vec-doc-relevance-training
└─ data
     └─ Split_Dataset
          ├─ Data
          │  ├─ train.npy
          │  ├─ test.npy
          |  └─ valid.npy
          └─ Ground_truth
             ├─ train.tsv
             ├─ test.tsv
             └─ valid.tsv
```

### Step 4: Optimization Pipeline

This pipeline aims to optimize hyperparameters for a fastText model using Optuna, train the model with the optimal parameters, and evaluate its performance using precision at N (Precision@N) and normalized discounted cumulative gain (NDCG) metrics.

Pipeline Steps:
+ Hyperparameter Optimization: Utilizes Optuna to search for the best hyperparameters for the fastText model.
+ Model Training: Trains the fastText model with the optimal hyperparameters using 80% of the training split data.
+ Embedding Generation: Generates embeddings for the remaining 20% of the test split data using the trained model.
+ Cosine Similarity Computation: Calculates cosine similarities for the generated embeddings.
+ Precision@N Calculation: Computes Precision@N scores, a measure of the relevance of retrieved documents, for the obtained cosine similarities.
+ NDCG Score Calculation: Computes normalized discounted cumulative gain (NDCG) scores, which assesses the quality of ranked search results based on relevance assessments.

In order to start the pipeline execution use this script, and run the following command:

``` 
python3 code/main.py [-i INPUT_TRAIN_FILE] [-t TEST_FILE] [-v VALIDATION_FILE] [-gt TEST_GROUND_TRUTH_FILE] [-gv VALIDATION_GROUND_TRUTH_FILE]  [-u USE_PRE_TRAINED_MODEL] [-c NO_OF CLASSES] [-win WINDOWS/LINUX]
 ``` 

You must pass the following four arguments:

+ -i/ --input : File path to the RELISH Train split dataset (.npy file format).
-t/ --test : File path to the RELISH Test split dataset (.npy file format).
-v/ --valid: File path to the RELISH Validation split dataset (.npy file format).
-gt/ --test_ground_truth : File path for the Test split ground truth (.tsv file format).
-gv/ --valid_ground_truth : File path for the Validation split ground truth (.tsv file format).
-u/ --use_pretrained : This is an optional parameter whether to use a pretrained fastText model. 1 - if yes; 0 - if no.
+ -c/  --classes : No. of classes to perform optimization on (Integer 2 or 3/ Default value is 3)
+ -win/ --windows : 1- if using Windows systems; 0- if using Unix-like systems (including Ubuntu)

To run this script, please execute the following command:

``` 
python3 code/main.py -i data/Split_Dataset/Data/train.npy -t data/Split_Dataset/Data/test.npy -v data/Split_Dataset/Data/valid.npy -gt data/Split_Dataset/Ground_truth/test.tsv -gv data/Split_Dataset/Ground_truth/valid.tsv -c 3 -win 0
 ``` 

Precision@N and NDCG scores are saved to TSV files in the following folder path: \output_2 (2 classes) and \output_3 (3 classes) for further analysis and reporting.

Make sure to run the model training twice for both the class distributions by changing the value of the -c/ --classes flag to 2 and 3.