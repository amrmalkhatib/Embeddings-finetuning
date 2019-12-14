import gensim.models as g
import numpy as np
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='Pre-trained embeddings fine-tuning')
parser.add_argument('-d', '--data', type=str, help='File path for tuning data')
parser.add_argument('-e', '--embeddings', type=str, help='File path for pretrained embeddings in word2vec_format')
parser.add_argument('-o', '--output_path', type=str, help='Fine-tuned embeddings output path')
parser.add_argument('-text', '--text_col', type=str, help='The name of the text column in the data file')
parser.add_argument('-label', '--label_col', type=str, help='The name of the label column in the data file')
parser.add_argument('-del', '--delimiter', type=str, default='\t', help="Data file's delimiter")


def load_data(tuning_data, text_column, label_column, data_delimiter):

    df = pd.read_csv(tuning_data, delimiter=data_delimiter)
    tune_data = df.reset_index()[[text_column, label_column]].values.tolist()
    return tune_data


if __name__ == "__main__":
    np.random.seed(7)
    embedding_dim = 300
    learning_rate = 0.0003
    num_epochs = 10
    min_word_count = 0
    context = 4  # Context window size

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    tune_data = load_data(args.data, args.text_col, args.label_col, args.delimiter)

    print("loading pretrained word vectors")

    pretrainedVectors = args.embeddings
    print("Fine-tuning word vectors")
    docs = [g.doc2vec.TaggedDocument(tweet[0].strip('"\'').lower().split(' '), tweet[1].strip()) for tweet in tune_data]
    doc2VecModel = g.Doc2Vec(docs, size=embedding_dim, window=context, min_count=min_word_count, sample=1e-5, workers=1,
                             hs=0, dm=0, negative=5, dbow_words=1, dm_concat=1, pretrained_emb=pretrainedVectors, iter=num_epochs)
    print("saving fine-tuned embeddings")
    doc2VecModel.save_word2vec_format(args.output_path)