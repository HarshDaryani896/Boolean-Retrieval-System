#!/usr/bin/env python
# coding: utf-8

# ## IR Assignment-1 (Boolean Information Retrieval System)
#  Harsh Daryani   2018B1A70645H                               
#  Rohan Sachan    2018B3A70992H                                
#  Aaryan Gupta    2018B1A70775H

# In[1]:


import nltk
import collections
import string
import os
import timeit
import math
from binarytree import Node


# ## Stopwords input from text file

# In[2]:


my_file = open(r"stopwords.txt", "r")
data = my_file.read()
stopwords_list = data.split("\n")
print("Number of stopwords: ",len(stopwords_list))
print(stopwords_list)
my_file.close()


# ## Index for Corpus Documents

# In[12]:


path=r"corpus"
docID_list = {i+1:doc for i, doc in enumerate(os.listdir(path))}
docID_list


# ## Preprocessing, Creation of Inverted Index and Parsing Query 

# In[13]:


#Initializing lists for tokens stemmed and unstemmed words
tokens_words_stemmed=[]
tokens_words_unstemmed=[]

#Initializing dictionary for word:frequency pair for unstemmed words
#To be utilized in spelling checker
unstemmed_dict={}


# In[14]:


class IRSystem():

    def __init__(self, docs=None, stop_words=stopwords_list):
        if docs is None:
            raise UserWarning('No Docs')
        self._docs = docs
        self._stemmer = nltk.stem.porter.PorterStemmer()
        self._inverted_index = self._preprocess_corpus(stop_words)
        self._inverted_index1 = self._preprocess_corpus1(stop_words)
        self._print_inverted_index()

    def _preprocess_corpus1(self, stop_words=stopwords_list):
        index = {}
        for i, doc in enumerate(self._docs):
            for word in doc.split():
                    #print(word) #prints all words from 1 to 42 docs, docs in alphanumerical order of name
                    token = word.lower()
                    if ((len(token)<40) and token.isnumeric()==False ):
                        if index.get(token, -244) == -244:
                            index[token] = Node(i + 1)
                        elif isinstance(index[token], Node):
                            index[token].insert(i + 1)
                        else:
                            raise UserWarning('Wrong data type for posting list')
        return index
    
    def _preprocess_corpus(self, stop_words=stopwords_list):
        index = {}
        for i, doc in enumerate(self._docs):
            for word in doc.split():
                    #print(word) #prints all words from 1 to 42 docs, docs in alphanumerical order of name
                    token = self._stemmer.stem(word.lower())
                    if ((token not in stop_words) and (len(token)<40) and token.isnumeric()==False ):
                        if index.get(token, -244) == -244:
                            index[token] = Node(i + 1)
                        elif isinstance(index[token], Node):
                            index[token].insert(i + 1)
                        else:
                            raise UserWarning('Wrong data type for posting list')
        return index
    
    def _print_inverted_index(self):
        print('UNSTEMMED INVERTED INDEX:\n')
        for word, tree in self._inverted_index1.items():
            tokens_words_unstemmed.append(word)
            unstemmed_dict[word]=len([doc_id for doc_id in tree.tree_data() if doc_id != None ])
            print('{}: {}'.format(word, [doc_id for doc_id in tree.tree_data() if doc_id != None ]))
        print()
        
        print('PREPROCESSED INVERTED INDEX:\n')
        for word, tree in self._inverted_index.items():
            tokens_words_stemmed.append(word)
            print('{}: {}'.format(word, [doc_id for doc_id in tree.tree_data() if doc_id != None]))
        print()


    def _get_posting_list(self, word):
        return [doc_id for doc_id in self._inverted_index[word].tree_data() if doc_id != None]

    @staticmethod
    def _parse_query(infix_tokens):
        precedence = {}
        precedence['NOT'] = 3
        precedence['AND'] = 2
        precedence['OR'] = 1
        precedence['('] = 0
        precedence[')'] = 0    

        output = []
        operator_stack = []

        for token in infix_tokens:
            if (token == '('):
                operator_stack.append(token)
            
            # if right bracket, pop all operators from operator stack onto output until we hit left bracket
            elif (token == ')'):
                operator = operator_stack.pop()
                while operator != '(':
                    output.append(operator)
                    operator = operator_stack.pop()
            
            # if operator, pop operators from operator stack to queue if they are of higher precedence
            elif (token in precedence):
                # if operator stack is not empty
                if (operator_stack):
                    current_operator = operator_stack[-1]
                    while (operator_stack and precedence[current_operator] > precedence[token]):
                        output.append(operator_stack.pop())
                        if (operator_stack):
                            current_operator = operator_stack[-1]
                operator_stack.append(token) # add token to stack
            else:
                output.append(token.lower())

        # while there are still operators on the stack, pop them into the queue
        while (operator_stack):
            output.append(operator_stack.pop())

        return output

    def process_query(self, query):
        # prepare query list
        query = query.replace('(', '( ')
        query = query.replace(')', ' )')
        query = query.split(' ')

        indexed_docIDs = list(range(1, len(self._docs) + 1))

        results_stack = []
        postfix_queue = collections.deque(self._parse_query(query)) # get query in postfix notation as a queue

        while postfix_queue:
            token = postfix_queue.popleft()
            result = [] # the evaluated result at each stage
            # if operand, add postings list for term to results stack
            if (token != 'AND' and token != 'OR' and token != 'NOT'):
                token = self._stemmer.stem(token) # stem the token
                # default empty list if not in dictionary
                if (token in self._inverted_index):
                    result = self._get_posting_list(token)
            
            elif (token == 'AND'):
                right_operand = results_stack.pop()
                left_operand = results_stack.pop()
                result = BooleanModel.and_operation(left_operand, right_operand)   # evaluate AND

            elif (token == 'OR'):
                right_operand = results_stack.pop()
                left_operand = results_stack.pop()
                result = BooleanModel.or_operation(left_operand, right_operand)    # evaluate OR

            elif (token == 'NOT'):
                right_operand = results_stack.pop()
                result = BooleanModel.not_operation(right_operand, indexed_docIDs) # evaluate NOT
            
            results_stack.append(result)                        
        if len(results_stack) != 1: 
            print("ERROR: Invalid Query. Please check query syntax.") # check for errors
            return None
        
        return results_stack.pop()


# ## Boolean Operations Handling

# In[15]:


class BooleanModel():
    
    @staticmethod
    def and_operation(left_operand, right_operand):
        # perform 'merge'
        result = []                                 # results list to be returned
        l_index = 0                                 # current index in left_operand
        r_index = 0                                 # current index in right_operand
        l_skip = int(math.sqrt(len(left_operand)))  # skip pointer distance for l_index
        r_skip = int(math.sqrt(len(right_operand))) # skip pointer distance for r_index

        while (l_index < len(left_operand) and r_index < len(right_operand)):
            l_item = left_operand[l_index]  # current item in left_operand
            r_item = right_operand[r_index] # current item in right_operand
            
            # case 1: if match
            if (l_item == r_item):
                result.append(l_item)   # add to results
                l_index += 1            # advance left index
                r_index += 1            # advance right index
            
            # case 2: if left item is more than right item
            elif (l_item > r_item):
                # if r_index can be skipped (if new r_index is still within range and resulting item is <= left item)
                if (r_index + r_skip < len(right_operand)) and right_operand[r_index + r_skip] <= l_item:
                    r_index += r_skip
                # else advance r_index by 1
                else:
                    r_index += 1

            # case 3: if left item is less than right item
            else:
                # if l_index can be skipped (if new l_index is still within range and resulting item is <= right item)
                if (l_index + l_skip < len(left_operand)) and left_operand[l_index + l_skip] <= r_item:
                    l_index += l_skip
                # else advance l_index by 1
                else:
                    l_index += 1

        return result

    @staticmethod
    def or_operation(left_operand, right_operand):
        result = []     # union of left and right operand
        l_index = 0     # current index in left_operand
        r_index = 0     # current index in right_operand

        # while lists have not yet been covered
        while (l_index < len(left_operand) or r_index < len(right_operand)):
            # if both list are not yet exhausted
            if (l_index < len(left_operand) and r_index < len(right_operand)):
                l_item = left_operand[l_index]  # current item in left_operand
                r_item = right_operand[r_index] # current item in right_operand
                
                # case 1: if items are equal, add either one to result and advance both pointers
                if (l_item == r_item):
                    result.append(l_item)
                    l_index += 1
                    r_index += 1

                # case 2: l_item greater than r_item, add r_item and advance r_index
                elif (l_item > r_item):
                    result.append(r_item)
                    r_index += 1

                # case 3: l_item lower than r_item, add l_item and advance l_index
                else:
                    result.append(l_item)
                    l_index += 1

            # if left_operand list is exhausted, append r_item and advance r_index
            elif (l_index >= len(left_operand)):
                r_item = right_operand[r_index]
                result.append(r_item)
                r_index += 1

            # else if right_operand list is exhausted, append l_item and advance l_index 
            else:
                l_item = left_operand[l_index]
                result.append(l_item)
                l_index += 1

        return result

    @staticmethod
    def not_operation(right_operand, indexed_docIDs):
        # complement of an empty list is list of all indexed docIDs
        if (not right_operand):
            return indexed_docIDs
        
        result = []
        r_index = 0 # index for right operand
        for item in indexed_docIDs:
            # if item do not match that in right_operand, it belongs to compliment 
            if (item != right_operand[r_index]):
                result.append(item)
            # else if item matches and r_index still can progress, advance it by 1
            elif (r_index + 1 < len(right_operand)):
                r_index += 1
        return result


# ## Inverted Index 

# In[16]:


path = r"corpus"
docs=[]
for root, dirs, files in sorted(os.walk(path)):
    for file in sorted(files):
        with open(os.path.join(path, file)) as f:
                docs.append(f.read().translate(str.maketrans('', '', string.punctuation)))



def main():
    ir = IRSystem(docs, stopwords_list)

    while True:
        query = input('Enter boolean query: ')
        query.translate(str.maketrans('', '', string.punctuation))
        start = timeit.default_timer()

        results = ir.process_query(query)
        
        stop = timeit.default_timer()

        if results is not None:
            print ('Processing time: {:.5} secs'.format(stop - start))
            print('\nDoc IDS: ')
            print(results)
        print()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        print('EXIT')


# In[17]:


#Stemmed words with preprocessing and stopwords excluded
print(tokens_words_stemmed)


# In[18]:


#Unstemmed words with preprocessing but stopwords included
print(tokens_words_unstemmed)


# ## Spelling Correction Query Handling

# In[19]:


def correct(word):
    "Find the best spelling correction for this word."
    # Prefer edit distance 0, then 1, then 2; otherwise default to word itself.
    candidates = (known(edits0(word)) or 
                  known(edits1(word)) or 
                  known(edits2(word)) or 
                  [word])
    return max(candidates, key=counts.get)

def known(words):
    "Return the subset of words that are actually in the dictionary."
    return {word for word in words 
                if word in counts}

def edits0(word): 
    "Return all strings that are zero edits away from word (i.e., just word itself)."
    return {word}

def edits2(word):
    "Return all strings that are two edits away from this word."
    return {e2 for e1 in edits1(word) 
                for e2 in edits1(e1)}

def edits1(word):
    "Return all strings that are one edit away from this word."
    pairs      = splits(word)
    deletes    = [a+b[1:]           for (a, b) in pairs if b]
    transposes = [a+b[1]+b[0]+b[2:] for (a, b) in pairs if len(b) > 1]
    replaces   = [a+c+b[1:]         for (a, b) in pairs for c in alphabet if b]
    inserts    = [a+c+b             for (a, b) in pairs for c in alphabet]
    return set(deletes + transposes + replaces + inserts)

def splits(word):
    "Return a list of all possible (first, rest) pairs that comprise word."
    return [(word[:i], word[i:]) 
                for i in range(len(word)+1)]


from string import ascii_lowercase as alphabet

assert alphabet == 'abcdefghijklmnopqrstuvwxyz'

import re
def tokens(text):
    "List all the word tokens (consecutive letters) in a text. Normalize to lowercase."
    return re.findall('[a-z]+', text.lower())

words = tokens_words_unstemmed
counts = unstemmed_dict
print(counts)


# In[20]:


phrase_uncorrected = input("Enter Query for Spelling Correction: ")
phrase_corrected = map(correct, tokens(phrase_uncorrected))

print("Original Token, ", "Correct Token", end="\n\n")
print(*zip(tokens(phrase_uncorrected), 
           phrase_corrected), sep="\n")

def correct_text(text):
    "Correct all the words within a text, returning the corrected text."
    return re.sub('[a-zA-Z]+', correct_match, text)

def correct_match(match):
    "Spell-correct word in match, and preserve proper upper/lower/title case."
    word = match.group()
    return case_of(word)(correct(word.lower()))

def case_of(text):
    "Return the case-function appropriate for text: upper, lower, title, or just str."
    return (str.upper if text.isupper() else
            str.lower if text.islower() else
            str.title if text.istitle() else
            str)

print(phrase_uncorrected)
print(correct_text(phrase_uncorrected))


# ## WildCard Query Handling

# In[21]:


#Rotate each word to create the permuterm index
def rotate(str, n):
    return str[n:] + str[:n]

# Create Permuterm Index
with open("permutermindex.txt","w") as f:
#     keys = tokens.keys()
    for token in sorted(tokens_words_unstemmed):
        dkey = token + "$"
        for i in range(len(dkey),0,-1):
            out = rotate(dkey,i)
            f.write(out)
            f.write(" ")
            f.write(token)
            f.write("\n")


# In[22]:


# Wildcard Query Types
#X*   *X    X*Y    X*Y*Z  *X* 


# In[25]:


def querying(query):
    final_result=[]
    queryA_list=[]
    queryB_list=[]
    
    # Split query and determine it's type
    parts = query.split("*")

    final_result.append('Query Processed as:-')
    final_result.append(parts)
    
    #These are the different cases formed depending on the number of wild card characters and their position in our query
    if len(parts)==1:
        case =0
    elif len(parts) == 3:
        case = 4
    elif parts[1] == "":
        case = 1
    elif parts[0] == "":
        case = 2
    elif parts[0] != "" and parts[1] != "":
        case = 3

    #Case 4 is dealt sperately as it has 2 sub queries    
    if case == 4:
        if parts[0] == "":
            case = 1

    # Read Permuterm Index
    permuterm = {}
    with open("permutermindex.txt") as f:
        for line in f:
            temp = line.split()
            permuterm[temp[0]] = temp[1]
    
    #This function will match the prefix of the word/wildcard query to the words in index
    
    def common_words(A,B):
        return set(A).intersection(B)
    
    def prefix_match(term, prefix):
        term_list = []
        for tk in term.keys():
            if tk.startswith(prefix):
                 #final_result.append(tk)     # Permuterm Index where wildcard query is matched
                term_list.append(term[tk])
        return term_list

    #This function is used to process query (ie after prefix match, the word and document is extracted where the prefix match has occured)
    def process_query(query):    
        term_list = prefix_match(permuterm,query)
        #print(term_list)
        final_result.append('Words Matching Wildcard Query:-')
        final_result.append(term_list)

    #Queries are processed on the bases of their cases
    if case == 0:
        pass
    elif case == 1:
        # 5) *X* = can be converted to X* form 
        if (parts[0]==''):
            query = parts[1]
            final_result.append('This is how the query will be processed')
            final_result.append(query)
        else:
            #1) X* = $X 
            query = "$" + parts[0]
            final_result.append('This is how the query will be processed')
            final_result.append(query)
    elif case == 2:
        # 2) *X = X$*
        query = parts[1] + "$"
        final_result.append('This is how the query will be processed')
        final_result.append(query)
    elif case == 3:
        # 3) X*Y = Y$X*
        query = parts[1] + "$" + parts[0]
        final_result.append('This is how the query will be processed')
        final_result.append(query)      
    elif case == 4:
        # 4) X*Y*Z = (Z$X*) and (Y*)
        queryA = parts[2] + "$" + parts[0]
        queryB = parts[1]
        final_result.append('This is how the query will be processed')
        final_result.append([queryA, queryB])

    if case != 4:
        process_query(query)
    elif case == 4:
        # 4) X*Y*Z = (Z$X*) and (Y*)
        
    # query A: Z$X*
        queryA_list = prefix_match(permuterm,queryA)
        final_result.append('This is out List contating the terms which match our desired queryA')
        final_result.append(queryA_list)
    # query B: Y*
        queryB_list= prefix_match(permuterm,queryB)
        final_result.append('This is out List contating the terms which match our desired queryB')
        final_result.append(queryB_list)  
        
    # Intersection of Query A and Query B words
        queryA_and_queryB = common_words(queryA_list,queryB_list)
        final_result.append('This is List contating common term documents for queryA and queryB')
        final_result.append(queryA_and_queryB)
    
    return(final_result)


# In[26]:


final_result= querying(input('Enter wildcard query: '))
final_result

