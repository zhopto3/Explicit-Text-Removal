"""Creates a lexicon of explicit and appropriate words from annotated data.

This script should be called using song metadata and lyrics in the csv format provided by MusicOSet (https://marianaossilva.github.io/DSW2019/).
It saves the lexicon to the filename/output path specified by the user in the command line. Words are classified as explicit
or appropriate based on work in a paper called "A hybrid modeling approach for an automated lyrics-rating system for adolescents"
by Kim and Yi (2019). 

Author:
    Zachary W. Hopton, 22-737-027

Example call:
    poetry run python hopton_project/make_lexicon.py --song-info "data/songs.csv" --lyrics "data/lyrics.csv" --output-path "data/lexicon.json"
"""

import argparse
import json
from collections import Counter
import re

import pandas as pd
import spacy
import numpy as np

nlp = spacy.load("en_core_web_md")

def getArgs()->argparse.ArgumentParser:
    parser = argparse.ArgumentParser("This is a program to create lexicons of explicit and appropriate terms.")

    parser.add_argument("--song-info", type = str, required = True,
                        help = "Path to csv file containing song metadata")
    parser.add_argument("--lyrics", type = str, required = True,
                        help = "Path to csv file containing lyrics of the songs in '--song-info'")
    parser.add_argument("--output-path", type = str, required = True,
                        help = "Path to output json file containing the lexicon.")
    return parser


class Lexicon:
    """A class used to construct Lexicon objects, which learn and represent information about words' classifications as explicit or appropriate.
    """

    def __init__(self, alpha: int = 20):
        self.counts = {}
        self.base_counts=Counter()
        self.scores = {"explicit":{},"approp":{}}
        self.sim_cache = {}
        #can be determined by user
        self.alpha = alpha
        #a variable that represents the cutoff scores for removal of negative words and insertion of positive words
        self.threshold = {"exp":0.,"apr":0.}

    def preprocess(self, song: str)->list:
        """A method to preprocess song lyrics 
            Parameters:
                song (str): String containing a song's lyrics
            Returns:
                Song_tokens (list): List of the song's tokens after meta data about verses and artists are removed
        """
        #tokenize and remove square bracket info here.
        song = song.lstrip('[').rstrip(']').lower()
        #delete meta info in brackets (chorus, pre-chorus, etc)
        song = re.sub(r'\[.*?\]','',song)
        #get rid of extra new char lines
        song = re.sub(r'\\n+',' ',song)
        #get rid of slashes before apostrophes
        song = song.replace('\\','')

        processed = nlp(song)

        return [tok.text for tok in processed]
    
    def _count(self, song: list, explicit: bool)->None:
        #update self.counts
        if explicit:
            for tok in song:
                if tok in self.counts:
                    self.counts[tok]["explicit"] += 1
                else:
                    self.counts[tok] = Counter()
                    self.counts[tok]["explicit"] += 1
        else:
            for tok in song:
                if tok in self.counts:
                    self.counts[tok]["approp"] += 1
                else:
                    self.counts[tok] = Counter()
                    self.counts[tok]["approp"] += 1     

    def _calc_logCDa(self, exp_count: float, approp_count: float)->float:
        """Calculate logCD alpha for the words present in the song lyrics
            Parameters:
                exp_count (float): Number of times a word appeared in an explicit song
                approp_count (float): Number of times a word appeared in an appropriate song 
            Returns:
                LogCDa (float): The value of the LogCD alpha for this word
        """
        if self.base_counts['explicit'] == 0 or self.base_counts['approp'] == 0:
            raise ZeroDivisionError("Data Invalid. Songs should be annotated as explicit or not")

        numerator = (exp_count+self.alpha)/self.base_counts['explicit']
        denominator = (approp_count+self.alpha)/self.base_counts['approp']

        return np.log(numerator/denominator)

    def combine_data(self, metadata: str, lyr: str)->pd.DataFrame:
        """Merge CSV files in order to associate song lyrics with explicit value
            Parameters:
                metadata (str): path to CSV containing metadata about songs in database
                lyr: (str): path to CSV containing lyrics of the song sin the database 
            Returns:
                Combined_Data (pd.DataFrame): metadata and lyrics CSV files merged on the column "song_id"
        """
        metaFrame = pd.read_csv(metadata, sep="\t", encoding="utf-8")
        metaFrame["song_name"]=metaFrame["song_name"].apply(lambda x: x.lower())
        lyrFrame = pd.read_csv(lyr, sep="\t", encoding="utf-8")
        return pd.merge(metaFrame, lyrFrame, on='song_id')

    def _initialize_counts(self, data: pd.DataFrame)->None:
        """A method to drive count collections necessary for calculating the LogCD alpha of each word in the data set
            Parameters:
                data (pd.DataFram): DataFrame containing song lyrics and "explicit" annotations
        """
        #get counts for the number of explicit and appropriate songs
        for song in data.itertuples(index = False, name = None):
            lyr = song[-1]
            #check that the lyrics field is not missing
            if type(lyr) == str:
                final_lyr = self.preprocess(lyr)
                #Check if explicit and count the words in each line
                if song[5]:  
                    self._count(final_lyr, True)
                    self.base_counts['explicit'] += 1 
                else:
                    self._count(final_lyr,False)
                    self.base_counts['approp'] += 1

    def get_scores(self, metadata: str, lyrics: str)->None:
        """Drive the processes necessary to fill a lexicon object with explicit and appropriate word dictionaries
            Parameters:
                metadata (str): path to CSV containing metadata about songs in database
                lyr: (str): path to CSV containing lyrics of the song sin the database 
        """
        #Align song metadata and lyrics according to song_id
        data = self.combine_data(metadata, lyrics)
        self._initialize_counts(data)

        #and now calculate the logCDalpha score for each word and update self.scores
        for tok in self.counts:
            score = self._calc_logCDa(self.counts[tok]["explicit"],self.counts[tok]["approp"])
            if score >= 0:
                self.scores['explicit'][tok] = score
            else:
                self.scores['approp'][tok] = score
   
    def update_thresh(self):
        """Calculate minimum explicit score to replace and maximum score to be considered appropriate"""
        assert len(self.scores["explicit"]) != 0, f"Before updating the threshold you must load a lexicon file or use the methods get_scores with new data"
        
        exp_scores = sorted(self.scores["explicit"].values())
        apr_scores = sorted(self.scores["approp"].values())
        #only remove most explicit 95%
        self.threshold["exp"] = np.percentile(exp_scores,5) # type: ignore
        #only add words that are in the 80% most appropriate
        self.threshold["apr"] = np.percentile(apr_scores,80) # type: ignore

    def find_most_similar(self, explicit: str)->str:
        """Find a similar but appropriate replacement for explicit words 
            Parameters:
                explicit (str): A word from the explicit dictionary that is to be replaced
            Returns:
                Most_similar (str): Most similar string in appropriate self.scores["approp"] that is below approp threshold
        """
        exp = nlp(explicit)
 
        most_similar_tok = (0,"")

        for word in self.scores["approp"].keys():
            if self.scores["approp"][word] < self.threshold["apr"]:

                app = nlp(word)

                if app.similarity(exp) > most_similar_tok[0]:
                    most_similar_tok = (app.similarity(exp), word)

        self.sim_cache[explicit] = most_similar_tok[1]
            
        return most_similar_tok[1]

    def load(self, lexicon: str)->None:
        """Load the self.scores and self.sim_cache from a previously saved lexicon object rather than getting scores on data again
            Parameters:
                lexicon (str): Path to a a previously saved lexicon (saved as json)
        """
        with open(lexicon, "r", encoding="utf-8") as input:
            self.scores, self.sim_cache = json.load(input)
        self.update_thresh()

    def save(self, output_name: str)-> None:
        """Save the self.scores and self.sim_cache to a json file so they can be reused
            Parameters:
                output_name (str): file path for (including file name) for saved lexicon
        """
        with open(output_name, "w", encoding="utf-8") as out:
            json.dump([self.scores,self.sim_cache], out, indent = 2)


def main():
    args = getArgs().parse_args()

    #create a lexicon obj, using default alpha level of 20
    lex = Lexicon()
    #get counts
    lex.get_scores(args.song_info, args.lyrics)

    #save counts
    lex.save(args.output_path)


if __name__ == "__main__":
    main()