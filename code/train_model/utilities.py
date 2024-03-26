import tqdm
import gensim
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from gensim.models import FastText
from typing import Union, List
import logging

log_file = 'fastText_utilities.log'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# Retrieves cleaned data from RELISH and TREC npy files
def process_data_from_npy(file_path_in: str = None) -> Union[List[str], List[List[str]], List[List[str]], List[List[str]]]:
    """
    Retrieves cleaned data from RELISH and TREC npy files, separating each column 
    into their own respective list.

    Parameters
    ----------
    filepathIn: str
            The filepath of the RELISH or TREC input npy file.
    Returns
    -------
    pmids: List[str]
            A list of all pubmed ids in the corpus.
    titles: List[List[str]]
            A list of lists where each sub-list contains the words 
            in the cleaned/processed title.
    abstracts: List[List[str]]
            A list of lists where each sub-list contains the words 
            in the cleaned/processed abstract.
    docs: List[List[str]]
            A list of lists where each sub-list contains the words 
            in the cleaned/processed document (title + abstract).
    """
    doc = np.load(file_path_in, allow_pickle=True)

    pmids = []
    docs = []

    for line in doc:
        pmids.append(line[0])
        if type(line[1]) == str:
            title_content = line[1].strip('][').split(', ')
            title = ' '.join(title_content).replace("\'", "")
            title_tokens = title.split(" ")
        else:
            title_tokens = line[1]
            
        if type(line[2]) == str:
            abstract_content = line[2].strip('][').split(', ')
            abstract = ' '.join(abstract_content).replace("\'", "")
            abstract_tokens = abstract.split(" ")
        else:
            abstract_tokens = line[2]
        
        docs.append(title_tokens + abstract_tokens)
        
    return (pmids, docs)

def create_fasttext_model(pmids: List[str], docs: List[List[str]], params: dict) -> FastText:
    """
    Create and train the fastText model using Gensim for the documents 
    in the corpus.

    Parameters
    ----------
    pmids: List[str]
            A list of all pubmed ids in the corpus.
    docs: List[List[str]]
            A list of lists where each sub-list contains the words 
            in the cleaned/processed document (title + abstract).
    params: dict
            Dictionary containing the parameters for the fastText model.
    Returns
    -------
    model: fastText
            fastText model.
    """
    model = FastText(**params)
    model.build_vocab(docs)
    model.train(docs, total_examples=model.corpus_count, epochs=model.epochs)

    return model

def save_model(model: FastText, output_file: str) -> None:
    """
    Saves the fastText model.

    Parameters
    ----------
    model: fastText
            fastText model.
    output_file: str
            File path of the fastText model generated.
    """
    model.save(output_file)
    
def calculate_cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

def get_similarity_scores(input_relevance_matrix, embeddings, output_matrix_name):
    # Read Embeddings
    embeddings_df = pd.read_pickle(embeddings)

    logging.info("Embeddings DataFrame Loaded")
    
    # Read Relevance matrix
    column_names = ["PID1", "PID2", "Value"]
    relevance_matrix_df = pd.read_csv(input_relevance_matrix, sep="\t", names = column_names, skiprows=1)

    # Adds empty columns to the file to store similarity scores
    relevance_matrix_df["Cosine Similarity"] = ""
    
    embeddings_dict = {int(pmid): embedding for pmid, embedding in zip(embeddings_df['PID'], embeddings_df['Embedding'])}

    # Create a list of ref and assessed PMID pairs
    pmid_pairs = list(zip(relevance_matrix_df["PID1"], relevance_matrix_df["PID2"]))

    for ref_pmid, assessed_pmid in tqdm.tqdm(pmid_pairs, total=len(pmid_pairs), desc="Calculating Similarities"):
        try:
            ref_pmid_vector = embeddings_dict[ref_pmid]
            assessed_pmid_vector = embeddings_dict[assessed_pmid]
            if ref_pmid_vector is not None and assessed_pmid_vector is not None:
                cosine_similarity = round(calculate_cosine_similarity(ref_pmid_vector, assessed_pmid_vector), 4)
                relevance_matrix_df.loc[(relevance_matrix_df['PID1'] == ref_pmid) & (relevance_matrix_df['PID2'] == assessed_pmid), 'Cosine Similarity'] = cosine_similarity
            else:
                logging.info(f"One of the vectors is None for ({ref_pmid}, {assessed_pmid})")
        except KeyError as e:
            logging.info(f"\nKeyError: {e}, ref_pmid: {ref_pmid}, assessed_pmid: {assessed_pmid}")
            break

    print('Added similarity scores')
    
    # Saves the updated matrix 
    relevance_matrix_df.to_csv(output_matrix_name, index=False, sep="\t")
    logging.info('Saved matrix')

def create_document_embeddings(pmids: list, documents: list, model, output_dir_path: str) -> None:
    """
    Generates document embeddings from the generated fastText model.
    Parameters
    ----------
    accessions : list
        List of accession numbers.
    documents : list
        List of function comments.
    model : 
        Pretraine Fasttext model.
    output_dir_path: str
        File path for the generated embeddings.
    """
    document_embeddings = []

    for index in range(len(pmids)):
        embeddings_list = []
        for word in documents[index]:
            try:
                embeddings_list.append(model.wv[word])
            except:
                continue
        #  Generate document embeddings from word embeddings
        first = True
        document = []
        for embedding in embeddings_list:
            if first:
                for dimension in embedding:
                    document.append(0.0)
                first = False
            doc_dimension = 0
            for dimension in embedding:
                document[doc_dimension] += dimension
                doc_dimension += 1
        doc_dimension = 0
        for dimension in document:
            # Get the average of each dimension of the embeddings and store it in the document list
            document[doc_dimension] = (dimension / len(embeddings_list))
            doc_dimension += 1
        document_embeddings.append(document)
    save_embeddings_to_pickle(pmids, document_embeddings, output_dir_path)

def save_embeddings_to_pickle(pmids, embeddings_list, output_file):
    data = {"PID": pmids, "Embedding": embeddings_list}
    df = pd.DataFrame(data)
    df.to_pickle(output_file)
    print(f"Embeddings saved to {output_file}")