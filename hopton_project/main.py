"""Remove explicit words from songs in a database or from text in .txt files.

This script should be called with a json file containing a lexicon of explicit and appropriate words, which can
be made running make_lexicon.py. The user must enter either a path to a txt file or a song title in the command line call. 
By default, the argument "replace" is set to true, meaning explicit words will be replaced with appropriate words. If false, 
explicit words are replaced with asterisks ('*'). Song title inputs are not case sensitive.

Author:
    Zachary W. Hopton, 22-737-027

Example calls:
    poetry run python hopton_project/main.py --lexicon "data/lexicon0406.json" --replace "False"  --song-title "bad romance"

    poetry run python hopton_project/main.py --lexicon "data/lexicon0406.json" --text "mini_text.txt"
"""

import sys
import re
import argparse
from collections import Counter

import pandas as pd
import spacy
from sacremoses import MosesDetokenizer

from make_lexicon import Lexicon

nlp = spacy.load("en_core_web_md")
md = MosesDetokenizer("en")

def getArgs()->argparse.ArgumentParser:
    parser = argparse.ArgumentParser("This is a program to remove explicit words from song lyrics or a text file. --song-title and --text are mutually exclusive.")

    parser.add_argument("--song-info", type = str, default = "data/songs.csv",
                        help = "Path to csv file containing song metadata")
    parser.add_argument("--lyrics", type = str, default = "data/lyrics.csv",
                        help = "Path to csv file containing lyrics of the songs in '--song-info'")
    parser.add_argument("--lexicon", type=str, required=True,
                        help = "Path to lexicon; If none are trained, train one using make_lexicon.py")
    parser.add_argument("--replace", type = str, default="True", choices = ("True","False"),
                       help="If True, explicit words will be replaced with similar, appropriate words")
    input = parser.add_mutually_exclusive_group(required=True)
    input.add_argument("--song-title", type = str,
                       help = "Song title for song you would like to clean")
    input.add_argument("--text", type = str,
                       help = "Path to a .txt file you would like to clean")
    
    return parser


def get_lyrics(data: pd.DataFrame, song: str, lexicon:Lexicon)->list:
    """Reteieve song lyrics from data
        Parameters:
            data (pd.DataFrame): Combination of the metadata and lyrics csv files input through the command line
            song (str): Song title passed through the command line
            lexicon (Lexicon): Lexicon of appropriate and explicit scores passed through the command line
        Returns:
            Lyrics (list): Preprocessed list containing the tokenized song lyrics 
    """
    lyrics = data.loc[data["song_name"]==song.lower(),"lyrics"].to_list()

    try:
        assert len(lyrics) == 1
    except AssertionError:
        print("This song title is ambiguous or not in the database you provided")
        sys.exit(0)
    
    return lexicon.preprocess(lyrics[0])


def preprocess_text(file_path: str)->list:
    """ Retrieve text from input text files and tokenize it
        Parameters:
            file_path (str): Path to text file to be cleaned, input through command line
        Returns:
            Text (list): List containing the tokenized song lyrics 
    """
    clean_text = []

    with open(file_path,"r",encoding="utf-8") as text: 
        for line in text:
            line = line.lower()
            processed = nlp(line)
            for tok in processed:
                clean_text.append(tok.text)
    try:
        assert len(clean_text) > 0
    except AssertionError:
        print("Text file must contain text")
        sys.exit(0)

    return clean_text


def find_replacement(token: str, lex:Lexicon)->str:
    """ Retrieve the most similar substitute appropriate word for an explicit word
        Parameters:
            token (str): Explicit token that should be replaced
            lex (Lexicon): Lexicon of appropriate and explicit scores passed through the command line
        Returns:
            Appropriate_Token (str): Token from the appropriate dictionary that is most similar to the input token
    """
    if token in lex.sim_cache:
        return "$"+lex.sim_cache[token]+"$"
    else:
        return "$"+lex.find_most_similar(token)+"$"


def remove_exp(input: list, lex:Lexicon, min_score: float, replace:bool)->list:
    """ Remove explicit words form input text or song 
        Parameters:
            input (list): Preprocessed list of tokens from the input text file or song
            lex (Lexicon): Lexicon of appropriate and explicit scores passed through the command line
            min_score (float): The minimum explicit score to remove, determined by the update_threshold method in the Lexicon class
            replace (bool): If True, replaces explicit words with appropriate words. Otherwise, removes and puts asterisks instead.
        Returns:
            Appropriate_Text (list): The list of tokens in the text with explicit words removed or replaced.
    """ 
    clean_txt = []

    for tok in input:
        if tok in lex.scores["explicit"] and lex.scores["explicit"][tok]>=min_score:
            if replace:
                clean_txt.append(find_replacement(tok,lex))
            else:
                clean_txt.append("*"*len(tok))
        else:
            clean_txt.append(tok)

    return clean_txt


def normalize(output: list)->str:    
    """ Detokenizes list, including with contractions pieced together and capitalized "I" as a pronoun.
        Parameters:
            output (list): Output list of the function "remove_exp"
        Returns:
            Normalized_text (str): The detokenized, appropriate text or song put in by the user. 
    """
    normed = []

    #making an iterator to have more control over where I am in the output list
    output_iter = iter(output)
    init_len = len(output)
    i = 0

    contractions = ["'ll","n't",
                "'s","'t",
                "'m","'d",
                "'re","'ve"]

    while i < init_len-1:
        cur_tok = next(output_iter)
        cur_tok = re.sub(r"[‘’]","'",cur_tok)

        next_tok = next(output_iter)
        next_tok = re.sub(r"[‘’]","'",next_tok)

        #connect contractions
        if next_tok.lower() in contractions:
            normed.append(cur_tok+next_tok)
        else:
            #look for cases like "freakin'" and "gon'"
            if next_tok == "'" and re.search(r"(in\$?\b|gon)",cur_tok):
                normed.append(cur_tok+next_tok)
            #look for "'cause"
            elif cur_tok == "'" and next_tok.lower()=="cause":
                normed.append(cur_tok+next_tok) 
            else:
                normed.append(cur_tok)
                normed.append(next_tok) 
        i += 2
    
    #append final token
    if i < init_len:
        final_tok = next(output_iter)
        final_tok = re.sub(r"[‘’]","'",final_tok)
        if final_tok in contractions:
            normed[-1] = normed[-1] + final_tok
        else:
            normed.append(final_tok)

    normed_text = md.detokenize(normed)
    #capitalize i and remove spaces around dashes
    normed_text = re.sub(r"\bi ","I ",normed_text)
    normed_text = re.sub(r"\bi'","I'",normed_text)
    normed_text = re.sub(r" ([-–—]) ",r"\1",normed_text)
    #connect contractions
    return normed_text


def pretty_print(normed_text: str)->None:
    """ Adds regular line breaks to and prints the appropriate output to the user's console.
        Parameters:
            normed_text (str): The detokenized, appropriate text or song put in by the user.
    """
    line = ""

    for char in normed_text:
        if line.count(" ") >= 5:
            if char == " ":
                print(line)
                line = ""
            else:
                line+=char
        else:
            line+=char
    
    print(line)


def main():
    args = getArgs().parse_args()

    #make a lexicon obj with out input data
    lex = Lexicon()
    lex.load(args.lexicon)
    thresh = lex.threshold["exp"]
    #get combined song data
    df_songs=lex.combine_data(args.song_info,args.lyrics)

    if args.text:
        try:
            assert args.text[-4:] == ".txt"
        except AssertionError:
            print("Error: If you choose to clean a text, you must pass the path to a .txt as an argument")
        text = preprocess_text(args.text)
    else:
        text = get_lyrics(df_songs, args.song_title, lex)

    if args.replace == "True":
        clean = remove_exp(text,lex,thresh,True)
    else:
        clean = remove_exp(text,lex,thresh,False)
    
    normed = normalize(clean)
    pretty_print(normed)
    #Save lexicon in order to store the similar pairs we cached
    lex.save(args.lexicon)

if __name__ == "__main__":
    main()