# Semester Project: Removing and Replacing Explicit Content from Songs
Intermediate Methods and Programming in Digital Linguistics, Spring 2023
Zachary W. Hopton (22-737-027)

This project flexibly allows users to remove explicit words from song lyrics in a database or text files. 
Explicit words can be removed and replaced by asterisks (*) or replaced by appropriate words. 

## Requirements

- This project's dependencies are installable via poetry (https://python-poetry.org/) by moving into the outer folder "hopton_project" and using the following command in the terminal:

    poetry install

## Data 

Relevant data for the project is stored in the following folder: 

    hopton_project/data

The files `lyrics.csv` and `songs.csv` contain the lyrics and metadata (singers, genre, explicit, etc.) of over 20,000 songs from the project "MusicOSet" (https://marianaossilva.github.io/DSW2019/) used as the main source of data for finding explicit and appropriate words in this project. For convenience during testing, two files in the above folder contain the first 25 lines of each of these files (`test_lyr.csv` and `test_songs.csv`).

The data directory also contains some text files used during testing, as well as a pre-trained lexicon of explicit and appropriate words (`lexicon0406.json`)

## Building a Lexicon of Explicit and Appropriate words

Words can be automatically classified as "explicit" or "appropriate" based on the frequency with which they appear in songs annotated as explicit. This project implements one method for doing so described by Kim and Yi in their 2019 paper, "A Hybrid Modeling Approach for an Automated Lyrics-Rating System for Adolescents." 

As a starting point in this project, users should create their own lexicon of explicit and appropriate words using the Lexicon class implemented in the following file:

    hopton_project/hopton_project/make_lexicon.py

The above script can be called from the command line, passing the relevant data as arguments, to automatically create a Lexicon object. The script will also save the dictionary of appropriate and inappropriate words to the specifed output location as a json file. The lexicon that is included in the data folder, `lexicon0406.json`, was saved with the following command line call (from inside hopton_project)

    poetry run python hopton_project/make_lexicon.py --song-info "data/songs.csv" --lyrics "data/lyrics.csv" --output-path "data/lexicon.json"

## Removing Explicit Content 

Once a user has created and saved a lexicon of appropriate and explicit words (or using the json already provided), they can use the following python script to begin removing content from songs or text files:

    hopton_project/hopton_project/main.py

Here, users must specify either a  `song-title` or `--text` argument (but not both). Users can choose whether they prefer to have explicit tokens in their text replaced with asterisks(`--replace "False"`) or similar, appropriate words (`--replace "True"`). Below are some example calls:

    poetry run python hopton_project/main.py --lexicon "data/lexicon0406.json" --replace "False"  --song-title "bad romance"

    poetry run python hopton_project/main.py --lexicon "data/lexicon0406.json" --text "data/mini_text.txt"

    poetry run python hopton_project/main.py --lexicon "data/lexicon0406.json" --replace "True" --song-title "Thank u, Next"

    poetry run python hopton_project/main.py --lexicon "data/lexicon0406.json" --replace "False" --text "data/roman_holiday.txt"

As a comprehensive search of the appropriate lexicon is done when replacing explicit words with the most similar appropriate ones, `--replace "False"` can be fairly time consuming. As one means of increasing it's speed, Lexicon objects are also associated with a `sim_cache` attribute. Any time an explicit word is replaced with an appropriate one in `main.py`, the Lexicon's `sim_cache` attribute is updated with the explicit word and it's most similar appropriate replacement, so it can be more quickly accessed in future searches.

A Lexicon's `sim_cache` is saved into the json with the explicit and appropriate lexicons. The file `lexicon0406.json` has a cache with some word pairs already saved in it from prior use. 

## Testing

The following files contain unit tests implemented through the library `pytest`:

    hopton_project/tests/test_main.py

    hopton_project/tests/test_make_lexicon.py

All tests can be conducted using the following command, from the outer folder `hopton_project`:

    poetry run py.test

## Future Directions

In the future, I plan to carry on with this project, particulary to improve the function that replaces explicit words with inappropriate ones. I would like to continue to find ways to make this approach more efficient (as opposed to an exhaustive search), while also experimenting with other ways of finding similar words, such as with WordNet. 

I would also like to experiment more with the alpha parameter used to calculate words' "explicit" score, and determine if training the lexicon on higher order n-grams in addition to unigrams would help pick up on more context sensitive explicit content.  