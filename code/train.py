import os
import time
import argparse
import logging
import utilities as utilities

def run(best_params, args, save_model=False):

    # 1) Load the training data
    train_pmids, train_docs = utilities.process_data_from_npy(args.input)
    logging.info("Retrieved RELISH Cleaned Data")

    # 2) Train the model with 80% of the data and best parameters
    start = time.time()
    model = utilities.create_fasttext_model(train_pmids, train_docs, best_params)
    end = time.time()
    logging.info(f"Time taken to train the model: {end - start} seconds")
    logging.info("RELISH fastText Model Generated")
    logging.info("Model is being used.")

    # 3) Set the test data to be used based on tuning parameter
    dataset_type = "Test"
    data_file = args.test
    ground_truth = args.test_ground_truth

    # 4) Load the data from npy file
    pmids, docs = utilities.process_data_from_npy(data_file)
    logging.info(f"Retrieved RELISH Cleaned {dataset_type} Data")

   # 5) Generate the embeddings: pd.DataFrame for loaded docs
    embeddings_df = utilities.create_document_embeddings(pmids, docs, model)
    logging.info(f"RELISH {dataset_type} Embeddings Pickle File Generated.")

    # 6) Generate the cosine similarity matrix: pd.DataFrame for the generated embeddings
    similarity_df = utilities.get_similarity_scores(ground_truth, embeddings_df)
    logging.info(f"RELISH {dataset_type} Cosine Similarity Matrix Generated.")

    # 7) Save the dataframes to a file each
    embeddings_file = f"output_{args.classes}/embeddings/test_embeddings_{args.classes}.pkl"
    similarity_file = f"output_{args.classes}/evaluation/test_cosine_similarity_{args.classes}.tsv"
    utilities.save_embeddings_to_pickle(embeddings_df, embeddings_file)
    utilities.save_similarity_to_tsv(similarity_df, similarity_file)

    # 8) Save the model in the given path if specified
    if save_model:
        model_file = f"output_{args.classes}/model/fastText_model_{args.classes}"
        utilities.save_model(model, model_file)

    return similarity_df, embeddings_df, model