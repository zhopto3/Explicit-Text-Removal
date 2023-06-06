"""Unit tests for functions in hopton_project/main.py.

Author:
    Zachary W. Hopton, 22-737-027
"""

import pytest
import sys
import re

sys.path.append("..")
sys.path.append("hopton_project")

from hopton_project.make_lexicon import Lexicon
import hopton_project.main

test_text="data/roman_holiday.txt"
mini_text = "data/mini_text.txt"
empty_text = "data/empty.txt"

#Create a reusable lexicon to pass to each of the unit tests
@pytest.fixture()
def test_lex():
    lex = Lexicon()
    lex.get_scores("data/test_songs.csv","data/test_lyr.csv")
    lex.update_thresh()
    return lex


def test_get_lyrics(test_lex):
    combined_data = test_lex.combine_data("data/test_songs.csv","data/test_lyr.csv")
    #test what happens with songs that are not in the data base
    with pytest.raises(SystemExit):
        hopton_project.main.get_lyrics(combined_data,"roman holiday",test_lex)
    
    #Check that varying capitalization yields the correct song
    assert hopton_project.main.get_lyrics(combined_data,"THANK U, NEXT", test_lex)[:8] == ["'","thought","i","'d","end","up","with","sean"]
    assert hopton_project.main.get_lyrics(combined_data,"thank u, next", test_lex)[:8] == ["'","thought","i","'d","end","up","with","sean"]


def test_preprocess_text():
    #Check that text is tokenized as expected
    assert len(hopton_project.main.preprocess_text(mini_text)) == 30
    #Check that empty test files result in errors
    with pytest.raises(SystemExit):
        hopton_project.main.preprocess_text(empty_text)
    #Check that casing is removed
    assert hopton_project.main.preprocess_text(mini_text)[1] == "chorus"
    assert hopton_project.main.preprocess_text(mini_text)[8] == "roman"


def test_find_replacement(test_lex):
    test_lex.sim_cache = {"cheese": "Test Results 1",
                      "Switzerland": "Test Results 2"}
    #Confirm that the cache is being used as expected
    assert hopton_project.main.find_replacement("cheese", test_lex) == "$Test Results 1$"
    assert hopton_project.main.find_replacement("Switzerland", test_lex) == "$Test Results 2$"

    #Check that vocabulary words out of the cache are returning a string from appropiate word lexicon
    assert hopton_project.main.find_replacement("damn", test_lex).strip("$") in test_lex.scores["approp"]
    #Check that the cache is updated with new words  
    assert "damn" in test_lex.sim_cache
    

def test_remove_exp_song(test_lex): 
    combined_data = test_lex.combine_data("data/test_songs.csv","data/test_lyr.csv")
    test_song = hopton_project.main.get_lyrics(combined_data,"thank u, next", test_lex)
    test_song2 = hopton_project.main.get_lyrics(combined_data, "breathin", test_lex)

    #Establish explicits are being removed as intended with song data
    for i, clean_tok in enumerate(hopton_project.main.remove_exp(test_song, test_lex,test_lex.threshold["exp"],False)): 
        #Assert that any explicit words in the output are below the threshold for inclusion
        if clean_tok in test_lex.scores["explicit"]:
            assert test_lex.scores["explicit"][clean_tok] < test_lex.threshold["exp"]
        if re.search(r"\*+", clean_tok):
            assert len(clean_tok) == len(test_song[i])

    for i, clean_tok in enumerate(hopton_project.main.remove_exp(test_song2, test_lex,test_lex.threshold["exp"],True)): 
        #Ensure replacement is being done instead of using asteriks
        assert re.search(r"\*+", clean_tok) == None
        if clean_tok[0] == "$" and clean_tok[-1] == "$":
            #Since words spacy model doesn't know can't be replaced by anything meaningful with this method, ignore
            if clean_tok != "$$":
                #Check that words being used to replace explicit words are appropriate
                assert clean_tok.strip("$") in test_lex.scores["approp"]
            #Confirm that any words being replaced were explicit
            assert test_song2[i] in test_lex.scores["explicit"]


def test_remove_exp_text(test_lex):
    #Now establish the same functions works the same with preprocessed text instead of songs
    test_file = hopton_project.main.preprocess_text(test_text)

    for i, clean_tok in enumerate(hopton_project.main.remove_exp(test_file, test_lex,test_lex.threshold["exp"],False)): 
        #Assert that any explicit words in the output are below the threshold for inclusion
        if clean_tok in test_lex.scores["explicit"]:
            assert test_lex.scores["explicit"][clean_tok] < test_lex.threshold["exp"]
        if re.search(r"\*+", clean_tok):
            assert len(clean_tok) == len(test_file[i])

    for i, clean_tok in enumerate(hopton_project.main.remove_exp(test_file, test_lex,test_lex.threshold["exp"],True)): 
        assert re.search(r"\*+", clean_tok) == None
        if clean_tok[0] == "$" and clean_tok[-1] == "$":
            if clean_tok != "$$":
                assert clean_tok.strip("$") in test_lex.scores["approp"]
            assert test_file[i] in test_lex.scores["explicit"]


def test_normalize():
    #Test that "I" is capitalized as intended
    test_caps = ["i","i","'m"]
    #Test that spaces around dashes (which the moses detokenizer adds) are removed
    test_hyph = ["hyphenated","-","phrase"]
    #Tests for spacing corrections in words containing apostrophes
    test_contractons = ["you","'d","'ve","we","'ll","I","’ll"]
    test_contract2 = ["and","'", "cause","we","'re","freakin","'","cool"]

    assert hopton_project.main.normalize(test_caps) == "I I'm"
    assert hopton_project.main.normalize(test_hyph) == "hyphenated-phrase"
    assert hopton_project.main.normalize(test_contractons) == "you'd've we'll I'll"
    assert hopton_project.main.normalize(test_contract2) == "and 'cause we're freakin' cool"


def test_pretty_print(capsys):
    FiveWordTest = "This is a six word sentence"
    SmallTest = "Now with four words"
    RemainderTest = "And now it's a sentence with eight words"

    hopton_project.main.pretty_print(FiveWordTest)
    output1 = capsys.readouterr()
    hopton_project.main.pretty_print(SmallTest)
    output2 = capsys.readouterr()
    hopton_project.main.pretty_print(RemainderTest)
    output3 = capsys.readouterr()

    #Tests to confirm that pretty_print splits lines as expected 
    assert output1.out == "This is a six word sentence\n"
    assert output2.out == "Now with four words\n"
    assert output3.out == "And now it's a sentence with\neight words\n"
