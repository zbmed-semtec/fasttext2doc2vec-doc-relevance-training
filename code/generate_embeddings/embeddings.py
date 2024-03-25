import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
import typing
from typing import Any, List, Iterable
from gensim.models import FastText
from gensim.models.fasttext import load_facebook_model

log_file = "fasttext.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')


def prepare_from_npy(filepathIn=None):
        '''
        Retrieves data from RELISH npy files, separating each column into their own respective list.

        Parameters
        ----------
        filepathIn: str
                The filepath of the RELISH or TREC input npy file.

        Returns
        -------
        list of str
                All pubmed ids associated to the paper.
        list of str
                All words within the title.
        list of str
                All words within the abstract.
        '''
        if not isinstance(filepathIn, str):
                logging.alert("Wrong parameter type for prepareFromTSV.")
                sys.exit("filepathIn needs to be of type string")
        else:
                doc = np.load(filepathIn, allow_pickle=True)
                pmids = []
                titles = []
                abstracts = []
                docs = []
                for line in doc:
                    pmids.append(int(line[0]))
                    if isinstance(line[1], (np.ndarray, np.generic)):
                        titles.append(np.ndarray.tolist(line[1]))
                        abstracts.append(np.ndarray.tolist(line[2]))
                        docs.append(np.ndarray.tolist(
                            line[1]) + np.ndarray.tolist(line[2]))
                    else:
                        titles.append(line[1])
                        abstracts.append(line[2])
                        docs.append(line[1] + line[2])
                return (pmids, titles, abstracts, docs)


def load_pretrained_model(model_filepath: str):
    """
    Loads the pre-trained model.
    Parameters
    ----------
    model_filepath : str
        Filepath of the downloaded pre-trained model.
    """    
    model = load_facebook_model(model_filepath)
    return model


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


    df = pd.DataFrame(list(zip((pmids), document_embeddings)), columns =['pmids', 'embeddings'])
    df = df.sort_values('pmids')
    df.to_pickle(output_dir_path)
    print("Embeddings Generated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Path to input RELISH tokenized .npy file")
    parser.add_argument("-p", "--pre_trained_model", type=int, default=None, help="Path to pre-trained model")               
    parser.add_argument("-o", "--output", type=str, help="Path to save embeddings pickle file")                 
    args = parser.parse_args()

    pmids, titles, abstracts, docs = prepare_from_npy(args.input)
    if args.pre_trained_model and os.path.isfile(args.pre_trained_model): 
        model = load_pretrained_model(args.model)
    else:
        params = {'sg': 0, 'vector_size':200, 'epochs':15, 'window':5, 'min_count':5, 'workers':8}
        model = create_fasttext_model(pmids, docs, params)
        save_model(model, "data/fasttext.model")
    create_document_embeddings(pmids, docs, model, args.output)
