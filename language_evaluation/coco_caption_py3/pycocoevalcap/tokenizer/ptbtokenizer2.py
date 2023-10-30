# #!/usr/bin/env python
# #
# # File Name : ptbtokenizer.py
# #
# # Description : Do the PTB Tokenization and remove punctuations.
# #
# # Creation Date : 29-12-2014
# # Last Modified : Thu Mar 19 09:53:35 2015
# # Authors : Hao Fang <hfang@uw.edu> and Tsung-Yi Lin <tl483@cornell.edu>

# import os
# import sys
# import subprocess
# import tempfile
# import itertools

# # path to the stanford corenlp jar
# STANFORD_CORENLP_3_4_1_JAR = 'stanford-corenlp-3.4.1.jar'

# # punctuations to be removed from the sentences
# PUNCTUATIONS = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
#         ".", "?", "!", ",", ":", "-", "--", "...", ";"]

# class PTBTokenizer:
#     """Python wrapper of Stanford PTBTokenizer"""
#     def __init__(self, tokenization_fn=None, verbose=True):
#         self.tokenization_fn = tokenization_fn
#         self.verbose = verbose

#     def  tokenize(self, captions_for_image):
#         cmd = ['java', '-cp', STANFORD_CORENLP_3_4_1_JAR, \
#                 'edu.stanford.nlp.process.PTBTokenizer', \
#                 '-preserveLines', '-lowerCase']

#         if self.tokenization_fn:
#             raise ValueError("Custom tokenizer is not supported yet")
#         else:
#             # ======================================================
#             # prepare data for PTB Tokenizer
#             # ======================================================
#             final_tokenized_captions_for_image = {}
#             image_id = [k for k, v in captions_for_image.items() for _ in range(len(v))]
#             sentences = '\n'.join([c['caption'].replace('\n', ' ') for k, v in captions_for_image.items() for c in v])

#             # ======================================================
#             # save sentences to temporary file
#             # ======================================================
#             path_to_jar_dirname=os.path.dirname(os.path.abspath(__file__))
#             tmp_file = tempfile.NamedTemporaryFile(delete=False, dir=path_to_jar_dirname)
#             tmp_file.write(sentences.encode())
#             tmp_file.close()

#             # ======================================================
#             # tokenize sentence
#             # ======================================================
#             cmd.append(os.path.basename(tmp_file.name))
#             if self.verbose:
#                 p_tokenizer = subprocess.Popen(cmd, cwd=path_to_jar_dirname, \
#                     stdout=subprocess.PIPE)
#             else:
#                 p_tokenizer = subprocess.Popen(cmd, cwd=path_to_jar_dirname, \
#                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
#             token_lines = p_tokenizer.communicate(input=sentences.rstrip())[0]
#             token_lines = token_lines.decode()
#             lines = token_lines.split('\n')
#             # remove temp file
#             os.remove(tmp_file.name)

#         # ======================================================
#         # create dictionary for tokenized captions
#         # ======================================================
#         for k, line in zip(image_id, lines):
#             if not k in final_tokenized_captions_for_image:
#                 final_tokenized_captions_for_image[k] = []
#             tokenized_caption = ' '.join([w for w in line.rstrip().split(' ') \
#                     if w not in PUNCTUATIONS])
#             final_tokenized_captions_for_image[k].append(tokenized_caption)

#         return final_tokenized_captions_for_image


#!/usr/bin/env python
#
# File Name : ptbtokenizer.py
#
# Description : Do the PTB Tokenization and remove punctuations.
#
# Creation Date : 29-12-2014
# Last Modified : Thu Mar 19 09:53:35 2015
# Authors : Hao Fang <hfang@uw.edu> and Tsung-Yi Lin <tl483@cornell.edu>

import os
import subprocess
import tempfile

class PTBTokenizer(object):
    """Python wrapper of Stanford PTBTokenizer"""

    corenlp_jar = 'stanford-corenlp-3.4.1.jar'
    punctuations = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
                    ".", "?", "!", ",", ":", "-", "--", "...", ";"]

    @classmethod
    def tokenize(cls, corpus):
        cmd = ['java', '-cp', cls.corenlp_jar, \
                'edu.stanford.nlp.process.PTBTokenizer', \
                '-preserveLines', '-lowerCase']

        if isinstance(corpus, list) or isinstance(corpus, tuple):
            if isinstance(corpus[0], list) or isinstance(corpus[0], tuple):
                corpus = {i:c for i, c in enumerate(corpus)}
            else:
                corpus = {i: [c, ] for i, c in enumerate(corpus)}

        # prepare data for PTB Tokenizer
        tokenized_corpus = {}
        image_id = [k for k, v in list(corpus.items()) for _ in range(len(v))]
        sentences = '\n'.join([c.replace('\n', ' ') for k, v in corpus.items() for c in v])

        # save sentences to temporary file
        path_to_jar_dirname=os.path.dirname(os.path.abspath(__file__))
        tmp_file = tempfile.NamedTemporaryFile(delete=False, dir=path_to_jar_dirname)
        tmp_file.write(sentences.encode())
        tmp_file.close()

        # tokenize sentence
        cmd.append(os.path.basename(tmp_file.name))
        p_tokenizer = subprocess.Popen(cmd, cwd=path_to_jar_dirname, \
                stdout=subprocess.PIPE, stderr=open(os.devnull, 'w'))
        token_lines = p_tokenizer.communicate(input=sentences.rstrip())[0]
        token_lines = token_lines.decode()
        lines = token_lines.split('\n')
        # remove temp file
        os.remove(tmp_file.name)

        # create dictionary for tokenized captions
        for k, line in zip(image_id, lines):
            if not k in tokenized_corpus:
                tokenized_corpus[k] = []
            tokenized_caption = ' '.join([w for w in line.rstrip().split(' ') \
                                          if w not in cls.punctuations])
            tokenized_corpus[k].append(tokenized_caption)

        return tokenized_corpus