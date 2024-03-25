import os
import time
import argparse
import logging
import utilities as utilities

log_file = "fastText_train.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def run(best_params, args, tuning):
    # Load the training data
    train_pmids, train_docs = utilities.process_data_from_npy(args.input)
    print("Retrieved RELISH Cleaned Data")
    logging.info("Retrieved RELISH Cleaned Data")

    start = time.time()
    # Train the model with 80% of the data and best parameters
    model = utilities.create_fasttext_model(train_pmids, train_docs, best_params)
    end = time.time()

    print(f"Time taken to train the model: {end - start} seconds")
    logging.info(f"Time taken to train the model: {end - start} seconds")
    print("RELISH fastText Model Generated")
    logging.info("RELISH fastText Model Generated")
    
    # Define a directory for storing models
    models_directory = f"models_{args.classes}"
    if not os.path.exists(models_directory):
        os.makedirs(models_directory)

    model_file = f"fastText_model_{int(time.time())}.pkl"
    model_file = os.path.join(models_directory, model_file)
    utilities.save_model(model, f"fasText_model_{int(time.time())}")
    logging.info("RELISH fastText Model Saved")

    print(model, "Model is being used.")

    if tuning == True:
        logging.info("Validation Split is being used")
        # use validation dataset for tuning
        test_pmids, test_docs = utilities.process_data_from_npy(args.valid)
    else:
        logging.info("Test Split is being used")
        # use test dataset for final evaluation
        test_pmids, test_docs = utilities.process_data_from_npy(args.test)
        
    print("Retrieved RELISH Cleaned Data")
    logging.info("Retrieved RELISH Cleaned Data")

    # Define a directory for storing embeddings
    embeddings_directory = f"embeddings_{args.classes}"
    if not os.path.exists(embeddings_directory):
        os.makedirs(embeddings_directory)

        
    embedding_file = f"embedding_{int(time.time())}.pkl"
    embeddings_file = os.path.join(embeddings_directory, embedding_file)

    # Generate the embeddings
    utilities.generate_embeddings(model, test_pmids, test_docs, embeddings_file)
    print("RELISH Embeddings Pickle File Saved")
    logging.info("RELISH Embeddings Pickle File Saved")

    # Define the directory for storing similarity results
    output_directory = f"output_{args.classes}"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Generate and save the cosine similarity matrix
    similarity_filename = f"cosine_similarity_{int(time.time())}.tsv"
    similarity_file = os.path.join(output_directory, similarity_filename)
    if tuning == True:
        logging.info("Validation Ground Truth is being used")
        utilities.get_similarity_scores(args.valid_ground_truth, embeddings_file, similarity_file)
    else:
        logging.info("Test Ground Truth is being used")
        utilities.get_similarity_scores(args.test_ground_truth, embeddings_file, similarity_file)

    print("RELISH Cosine Similarity Matrix Saved")
    logging.info("RELISH Cosine Similarity Matrix Saved")

    return similarity_file