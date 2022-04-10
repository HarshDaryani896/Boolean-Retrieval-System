*******************************************************
IR Assignment-1 (Boolean Information Retrieval System)
Harsh Daryani 2018B1A70645H
Rohan Sachan 2018B3A70992H
Aaryan Gupta 2018B1A70775H
*******************************************************

How to Run the code:

step 1: open the .ipynb file locally on your jupyter notebook.
step 2: can the path of stopword.txt and corpus accorking to your system (if they are not in the same folder as the .ipynb file)
step 3: Run the code using kernal > restart and run all
step 4: Enter the input(query) as and whereas asked by the code.

*******************************************************

Design Document:

Data structures use: Binary tree, dictionary, lists, sets, collections.

libraries used:
import nltk
import collections
import string
import os
import timeit
import math
from binarytree import Node
import re

We used B-tree for creating inverted index, dictionary for saving the unstemmed tokens with word:frequency for spelling check when the edit distance of 2 words is similar for different operations. It creates permuterm index and stores in the text file in the same location as .ipynb file. 
*******************************************************

Stopword Removal:

if needed you can add or remove words from stopwords.txt.

Stemming/Lemmatization:

for eg:
Flavius > Flaviu

it uses nltk porter stemmer algorithm.

Building Index

Inverted index used for each stemed word.

Querying

Text Cases:

*******************************************************
Boolean Query:
Enter boolean query: shakespeare OR juliet
Processing time: 0.0001525 secs

Doc IDS: 
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]

*******************************************************
Spelling Correction
Enter Query for Spelling Correction: sakespeare or julet
Original Token,  Correct Token

('sakespeare', 'shakespeare')
('or', 'or')
('julet', 'juliet')
sakespeare or julet
shakespeare or juliet

*******************************************************
Wildcard Query
Enter wildcard query: jul*
['Query Processed as:-',
 ['jul', ''],
 'This is how the query will be processed',
 '$jul',
 'Words Matching Wildcard Query:-',
 ['jul',
  'jule',
  'julia',
  'julias',
  'juliasay',
  'juliet',
  'julietfast',
  'juliets',
  'julietta',
  'juliettas',
  'julio',
  'julius',
  'july',
  'julys']]

*******************************************************