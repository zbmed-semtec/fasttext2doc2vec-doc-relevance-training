import os
import time
import argparse
import logging
import subprocess
import utilities as utilities
from gensim.models import FastText
from gensim.models.fasttext import load_facebook_model

def run_pretrained(args, model_directory):

    # 1) Downloaded pre-trained fastText model
    try:
        subprocess.run(['wget', 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz'], cwd=model_directory, check=True)
        logging.info("fastText Download completed.")

    except subprocess.CalledProcessError as e:
        logging.error(f"An error occurred during setup: {e}")
        return

    # 2) Load the pre-trained fastText model
    model_path = f'{model_directory}/cc.en.300.bin.gz'
    try:
        model = utilities.load_pretrained_model(model_path)
        logging.info("Pre-trained fastText model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load the pre-trained model: {e}")
        return

    # 3) Load the data from npy file
    test_pmids, test_docs = utilities.process_data_from_npy(data_file)
    logging.info(f"Retrieved RELISH Cleaned Test Data")

    # 4) Generate the embeddings: pd.DataFrame for loaded docs
    test_embeddings_df = utilities.create_document_embeddings(test_pmids, test_docs, model, args.pretrained)
    logging.info(f"RELISH Test Embeddings Pickle File Generated.")

    # 5) Generate the cosine similarity matrix: pd.DataFrame for the generated embeddings
    test_similarity_df = utilities.get_similarity_scores(args.test_ground_truth, test_embeddings_df)
    logging.info(f"RELISH Test Cosine Similarity Matrix Generated.")

    test_embeddings_file = f"output_{args.classes}/embeddings/best_embeddings_{args.classes}.pkl"
    test_similarity_file = f"output_{args.classes}/evaluation/best_cosine_similarity_{args.classes}.tsv"
    
    utilities.save_embeddings_to_pickle(test_embeddings_df, test_embeddings_file)
    utilities.save_similarity_to_tsv(test_similarity_df, test_similarity_file)