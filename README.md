# FastText2Doc2Vec-Doc-relevance

This repository focuses on an approach exploring and assessing literature-based doc-2-doc recommendations using the fastText algorithm with its application to the RELISH dataset. The workflow involves training the fastText models on a specified training set and then evaluating the document-to-document recommendations on a separate test set. Additionally, we employ Optuna for optimizing the hyperparameters for the trained fastText models.

## Table of Contents

1. [About](#about)
2. [Input Data](#input-data)
3. [Pipeline](#pipeline)
    1. [Generate Embeddings](#generate-embeddings)
    2. [Train and Optimize fastText models](#train-and-optimize-fasttext-models)
    3. [Cosine Similarity Computation](#cosine-similarity-computation)
    2. [Evaluation](#evaluation)
        - [Precision@N](#precisionn)
        - [nDCG@N](#ndcgn)
4. [Getting Started](#getting-started)

## About

Our approach employs the [FastText library](https://fasttext.cc/docs/en/support.htm) to generate word embeddings and subsequently employs a centroid aggregration technique to produce document-level embeddings. This process involves calculating the centroids of word embeddings found in the titles and abstracts of each document. The approach is applied to evaluate literature-based document-to-document recommendations using the RELISH dataset.

## Input Data
The input data for this method consists of preprocessed tokens derived from the RELISH documents. These tokens are stored in the RELISH.npy file, which contains preprocessed arrays comprising PMIDs, document titles, and abstracts. These arrays are generated through an extensive preprocessing pipeline, as elaborated in the [relish-preprocessing repository](https://github.com/zbmed-semtec/relish-preprocessing). Within this preprocessing pipeline, both the title and abstract texts undergo several stages of refinement: structural words are eliminated, text is converted to lowercase, stop words are removed and finally, tokenization is employed, resulting in arrays of individual words.

## Pipeline

This section outlines the progression from generating document embeddings to conducting hyperparameter optimization and ultimately evaluating the effectiveness of the approach.

### Generate Embeddings
The following section outlines the process of generating document-level embeddings for each PMID of the RELISH corpus using either the pre-trained fastText model or by training our own fastText models. We employ the parameters shown below in order to generate our models.

### Train and Optimize fastText models
We create and train fastText models with customizable hyperparameters to comprehend the connections between documents and words in a high-dimensional vector space. We aim to optimize these hyperparameters to establish the most effective relationship between cosine similarity and document relevance.

To accomplish this we begin by splitting the dataset into a training set and a testing set. The training set is then used to train the fastText model, where we explore various hyperparameters to optimize its performance. This optimization process is crucial for enhancing the model's ability to capture meaningful relationships between cosine similarity and document relevance. For each set of hyperparameters, a fastText model is trained on the training split.

Following this, we evaluate the model's performance on the testing set using Precision@5 as our evaluation metric.

##### Parameters

+ **sg:** {1,0} Refers to the training algorithm. If sg=1, skip-gram is used otherwise, continuous bag of words is used.
+ **vector_size:** It represents the dimensions of the generated embeddings, with options of 200, 300 and 400 in our case.
+ **window:** Represents the maximum distance between the current and predicted word, with values fof 5,6 and 7 in our case.
+ **epochs:** Refers to the number of iterations over the training dataseta and is set at 15 in this context.
+ **min_count:** It is the minimum number of appearances a word must have to not be ignored by the algorithm and is configured at a minimum of 5.

### Cosine Similarity Computation

Following hyperparameter optimization where the best model gets saved, embeddings are generated for the test dataset using this trained model. Subsequently, cosine similarity is calculated for the test dataset embeddings, providing a measure of similarity between pairs of documents based on their learned representations.

## Evaluation

### Precision@N

In order to evaluate the effectiveness of this approach, we make use of Precision@N. Precision@N measures the precision of retrieved documents at various cutoff points (N).We generate a Precision@N matrix for existing pairs of documents within the RELISH corpus, based on the original RELISH JSON file. The code determines the number of true positives within the top N pairs and computes Precision@N scores. The result is a Precision@N matrix with values at different cutoff points, including average scores. For detailed insights into the algorithm, please refer to this [documentation](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/Precision%40N_existing_pairs).


### nDCG@N

Another metric used is the nDCG@N (normalized Discounted Cumulative Gain). This ranking metric assesses document retrieval quality by considering both relevance and document ranking. It operates by using a TSV file containing relevance and cosine similarity scores, involving the computation of DCG@N and iDCG@N scores. The result is an nDCG@N matrix for various cutoff values (N) and each PMID in the corpus, with detailed information available in the [documentation](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/Evaluation).


## Getting Started

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


- Use the [Download_Data.sh](./Download_Data.sh) script to download the Split Dataset by running the following commands:

```
chmod +777 Download_Data.sh
./Download_Data.sh
```
This script makes sure that the necessary folders are created and the files are downloaded in the corresponding folders.

**OR**


- You could also download the dataset from this link: [Split_Dataset](https://drive.google.com/drive/folders/1Bq_U5207utn7tvSt_HLVdOdYR5QW7MMN). Please make sure to keep the data in the below specified format.

```
📦 /fasttext2doc2vec-doc-relevance-training
└─ data
   └─ Split_Dataset
      ├─ Data
      │  ├─ Train
      │  │  └─ relish_train_tokens_removed_stopwords.npy
      │  ├─ Test
      │  │  └─ relish_test_tokens_removed_stopwords.npy
      │  └─ Valid
      │     └─ relish_val_tokens_removed_stopwords.npy
      └─ Ground_truth
         ├─ train_split.tsv
         ├─ test_split.tsv
         └─ val_split.tsv
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
python3 code/train_model/main.py [-i INPUT] [-v VALIDATION_FILE] [-t TEST_FILE] [-gv VALIDATION_GROUND_TRUTH] [-gt TEST_GROUND_TRUTH] [-c NO_OF CLASSES] [-win WINDOWS/LINUX]
 ``` 

 You must pass the following four arguments:

+ -i/ --input : File path to the RELISH Train split dataset (.npy file format).
+ -v/ --valid : File path to the RELISH Validation split dataset (.npy file format).
+ -t/ --test : File path to the RELISH Test split dataset (.npy file format).
+ -gv/ --valid_ground_truth : File path for the Validation split ground truth (.tsv file format).
+ -gt/ --test_ground_truth : File path for the Test split ground truth (.tsv file format).
+ -c/ --classes : No. of classes to perform optimization on (Integer 2 or 3/ Default value is 3).
+ -win/ --windows : 1 - if using Windows systems; 0 - if using Unix-like systems (including Ubuntu)

To run this script, please execute the following command:

 ``` 
python3 code/train_model/main.py -i data/Split_Dataset/Data/train.npy -v data/Split_Dataset/Data/valid.npy -t data/Split_Dataset/Data/test.npy -gv data/Split_Dataset/Groundtruth/valid.tsv -gt data/Split_Dataset/Groundtruth/test.tsv -c 3 -win 0
 ``` 

Precision@N and NDCG scores are saved as TSV files in the following folder path: `\output_2\evaluation\`  for 2 class distribution and `\output_3\evaulation\` for 3 class distribution for further analysis and reporting.

Make sure to run the model training twice for both the class distributions by changing the value of the -c/ --classes flag to 2 and 3.

**NOTE:** As of now, we use the test file as our validation dataset during the model training. Make sure to replace the validation dataset with the truth dataset as well as validation groundtruth file with the test groundtruth file.

For replacing the validation data with the test data, please execute the following command:

``` 
python3 code/train_model/main.py -i data/Split_Dataset/Data/train.npy -v data/Split_Dataset/Data/test.npy -t data/Split_Dataset/Data/test.npy -gv data/Split_Dataset/Groundtruth/test.tsv -gt data/Split_Dataset/Groundtruth/test.tsv -c 3 -win 0
``` 