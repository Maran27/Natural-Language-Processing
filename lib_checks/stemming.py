# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 19:06:06 2021

@author: hp
"""

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

paragraph = '''I'm Taylor, good evening. I wanna first thank Billboard from the bottom of my heart for this honor…I wanna say thank you so much to Billboard for giving me this honor, for naming me as their Woman of the Decade. So what does it mean to be the woman of this decade? Well, it means I've seen a lot. When this decade began I was 20 years old and I had put out my self-titled debut album when I was 16, and then the album that would become my breakthrough album, which was called Fearless. And I saw that there was a world of music and experience beyond country music that I was really curious about.
I saw pop stations send my songs ‘Love Story’ and ‘You Belong With Me’ to number one for the first time. And I saw that as a female in this industry, some people will always have slight reservations about you. Whether you deserve to be there, whether your male producer or co-writer is the reason for your success, or whether it was a savvy record label. It wasn't. I saw that people love to explain away a woman's success in the music industry, and I saw something in me change due to this realization. This was the decade when I became a mirror for my detractors. Whatever they decided I couldn't do is exactly what I did….Whatever they criticized about me became material for musical satires or inspirational anthems, and the best lyrical examples I can think of are songs like ‘Mean,’ ‘Shake It Off,’ and ‘Blank Space.’ Basically if people had something to say about me, I usually said something back in my own way.'''

sentences = nltk.sent_tokenize(paragraph)
stemmer = PorterStemmer()

stop_words = stopwords.words('english')

for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))] #list comprehension
    sentences[i] = ' '.join(words)
    
    
