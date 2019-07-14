from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import argparse

# Parse command line arguments
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument(
    "text",
    help="Teks yang hendak diproses dengan tagger NER",
)

arguments = argument_parser.parse_args()

# Initialize tagger
tagger = StanfordNERTagger(
    "./ner-model.ser.gz",
    "./resources/stanford-ner.jar",
    encoding='utf-8'
)

tokens = word_tokenize(arguments.text)
tagged = tagger.tag(tokens)

print(tagged)