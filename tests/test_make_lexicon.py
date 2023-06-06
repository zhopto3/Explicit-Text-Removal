"""Unit tests for functions in hopton_project/make_lexicon.py.

Author:
    Zachary W. Hopton, 22-737-027
"""

import pytest

import numpy as np

from hopton_project.make_lexicon import Lexicon

#Create a reusable lexicon to pass to each of the unit tests
@pytest.fixture()
def test_lex():
    return Lexicon()


def test_combine_data(test_lex):
    output = test_lex.combine_data("data/test_songs.csv","data/test_lyr.csv")
    #Combined data should have same number of rows as the test datasets
    assert output.shape[0] == 24
    #Test that song metadata is in the expected location in new dataset 
    assert output.loc[1,"explicit"] == True
    assert output.loc[23,"song_id"] == "5ASM6Qjiav2xPe7gRkQMsQ"
    #Check that song title is lowered
    assert output.loc[0, "song_name"] == "thank u, next"
    #Ensure lyrics are aligned with the songs as expected
    assert output.loc[0, "lyrics"][-5:] == "yee']"
    assert output.loc[23,"lyrics"][-8:] == "Ding!)']"


def test_preprocess(test_lex):
    sample1 = "['[Verse 1]\nThought I\'d end up with Sean\nBut he wasn\'t a match\nWrote some songs about Ricky\nNow I listen"
    sample2 = "[I\'m won\'t can\'t]"
    sample3 = "[Verse 1 is [faster].]"

    #Test that preprocessing removes metadata and expected symbols, and results in token lists
    assert test_lex.preprocess(sample1) == ["'", '\n', 'thought', 'i', "'d", 'end', 'up', 'with', 'sean', '\n', 'but', 'he', 'was', "n't", 'a', 'match', '\n', 'wrote', 'some', 'songs', 'about', 'ricky', '\n', 'now', 'i', 'listen']
    assert test_lex.preprocess(sample2) == ["i","'m","wo","n't","ca","n't"]
    assert test_lex.preprocess(sample3) == ["verse","1","is","."]

def test_count(test_lex):
    sample1 = ["\n","do","n't","care","about","the","presents","\n","underneath","the","christmas","tree","\n"]
    sample2 = ["\n","did","n't","have","a","dime","but","i","always","had","a","vision"]
    sample3 = []
    
    test_lex._count(sample1, True)
    test_lex._count(sample2, False)
    test_lex._count(sample3,True)
    #Confirm that counts of types are being collected as expected
    assert len(test_lex.counts) == 19
    assert test_lex.counts["\n"]["approp"] == 1
    assert test_lex.counts["the"]["explicit"] == 2


def test_calc_logCDa(test_lex):
    test_lex.base_counts = {"explicit": 0, "approp":0}
    with pytest.raises(ZeroDivisionError):
        test_lex._calc_logCDa(0,0)
    
    test_lex.base_counts = {"explicit":25,"approp":25}
    #all else equal, a word appearing mostly in appropriate songs should have a negative score
    assert test_lex._calc_logCDa(0,40) < 0 
    #all else equal, a word appearing mostly in explicit songs should have a positive score
    assert test_lex._calc_logCDa(40,39) > 0
    test_lex.base_counts = {"explicit":2,"approp":5}
    assert test_lex._calc_logCDa(20, 5) == np.log(4)


def test_initialize_counts(test_lex):
    data = test_lex.combine_data("data/test_songs.csv","data/test_lyr.csv")
    test_lex._initialize_counts(data)

    #number of lyrics should be 23 since one song is missing lyrics
    assert test_lex.base_counts["explicit"]+test_lex.base_counts["approp"] == 23
    #ensure other counts have been added
    assert len(test_lex.counts.keys()) > 0


def test_get_scores(test_lex):
    test_lex.get_scores("data/test_songs.csv","data/test_lyr.csv")

    #confirm only positive log cd scores are classified as explicit
    for val in test_lex.scores["explicit"].values():
        assert val >= 0
    for val in test_lex.scores["approp"].values():
        assert val < 0


def test_update_thresh(test_lex):
    #ensure that assertion error is raised if the method is called while scores are empty
    with pytest.raises(AssertionError):
        print(test_lex.scores)
        test_lex.update_thresh()

    test_lex.get_scores("data/test_songs.csv","data/test_lyr.csv")
    test_lex.update_thresh()

    #ensure that a lexicon's threshold values are being updated from original value of 0.0
    assert test_lex.threshold["exp"] > 0
    assert test_lex.threshold["apr"] < 0


def test_find_most_similar(test_lex):
    test_lex.scores = {"explicit":{"stuff":0.5,
                                   "animal":1.5,
                                   "plant":1.25},
                       "approp":{"cow":-0.9,
                                 "fly":-0.2,
                                 "ham":-1.4}}
    test_lex.update_thresh()
    #ensure that the candidate is in the appropriate lexicon
    assert test_lex.find_most_similar("stuff") in test_lex.scores["approp"]
    #ensure score of ouput is below the threshold for appropriate words (more negative = more appropriate) 
    assert test_lex.scores["approp"][test_lex.find_most_similar("animal")] < test_lex.threshold["apr"]