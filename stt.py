import speech_recognition as sr
import pyttsx3
import cv2
from punctuator import Punctuator
import nltk
import re
import heapq
import numpy as np
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import networkx as nx
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from summa import keywords
from rake_nltk import Rake
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# from sentence_transformers import SentenceTransformer

def SpeakText(command):
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()
    
def textSpeech(speak):
    r = sr.Recognizer()
    MyText = ''
    k = 1
    while(k!=0):
        try:
            with sr.Microphone() as source2:
                r.adjust_for_ambient_noise(source2, duration=0.2)
                print('Say The Answer')
                audio2 = r.listen(source2)
                MyText+=r.recognize_google(audio2)
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
        except sr.UnknownValueError:
            print("unknown error occured")
        if cv2.waitKey(0):
            break
    if speak:
        SpeakText(MyText)
    return MyText

def punctuate(MyText):
    p = Punctuator('./punctuator/Demo-Europarl-EN.pcl')
    MyText = p.punctuate(MyText)
    return MyText

def summarize1(MyText):
    file_name = 'input.txt'
    output_location = 'summary.txt'
    sent_word_length = 15
    top_n = 3

    def read_text(file_name):
        """
        Read text from file
        INPUT:
        file_name - Text file containing original text.
        OUTPUT:
        text - str. Text with reference number, i.e. [1], [10] replaced with space, if any...
        clean_text - str. Lowercase characters with digits & one or more spaces replaced with single space.
        """
        # with open(file_name, 'r', encoding="utf8") as f:
        #     file_data = f.read()

        text = MyText
        text = re.sub(r'\[[0-9]*\]',' ',text)
        text = re.sub(r'\s+',' ',text)

        clean_text = text.lower()

        # replace characters other than [a-zA-Z0-9], digits & one or more spaces with single space
        regex_patterns = [r'\W',r'\d',r'\s+']
        for regex in regex_patterns:
            clean_text = re.sub(regex,' ',clean_text)

        return text, clean_text

    def rank_sentence(text, clean_text, sent_word_length):
        """
        Rank each sentence and return sentence score
        INPUT:
        text - str. Text with reference numbers, i.e. [1], [10] removed, if any...
        clean_text - str. Clean lowercase characters with digits and additional spaces removed.
        sent_word_length - int. Maximum number of words in a sentence.
        OUTPUT:
        sentence_score - dict. Sentence score
        """
        sentences = nltk.sent_tokenize(text)
        stop_words = nltk.corpus.stopwords.words('english')

        word_count = {}
        for word in nltk.word_tokenize(clean_text):
            if word not in stop_words:
                if word not in word_count.keys():
                    word_count[word] = 1
                else:
                    word_count[word] += 1

        sentence_score = {}
        for sentence in sentences:
            for word in nltk.word_tokenize(sentence.lower()):
                if word in word_count.keys():
                    if len(sentence.split(' ')) < int(sent_word_length):
                        if sentence not in sentence_score.keys():
                            sentence_score[sentence] = word_count[word]
                        else:
                            sentence_score[sentence] += word_count[word]

        return sentence_score

    def generate_summary(file_name, sent_word_length, top_n):
        """
        Generate summary
        INPUT:
        file_name - Text file containing original text.
        sent_word_length - int. Maximum number of words in a sentence.
        top_n - int. Top n sentences to display.
        OUTPUT:
        summarized_text - str. Summarized text with each sentence on each line.
        """
        text, clean_text = read_text(file_name)

        sentence_score = rank_sentence(text, clean_text, sent_word_length)

        best_sentences = heapq.nlargest(int(top_n), sentence_score, key=sentence_score.get)

        summarized_text = []

        sentences = nltk.sent_tokenize(text)

        for sentence in sentences:
            if sentence in best_sentences:
                summarized_text.append(sentence)

        summarized_text = "\n".join(summarized_text)

        return summarized_text

    summary = generate_summary(file_name, sent_word_length, top_n)

    return summary

def summarize2(MyText):
    def _create_frequency_table(text_string) -> dict:
        stopWords = set(stopwords.words("english"))
        words = word_tokenize(text_string)
        ps = PorterStemmer()
        freqTable = dict()
        for word in words:
            word = ps.stem(word)
            if word in stopWords:
                continue
            if word in freqTable:
                freqTable[word] += 1
            else:
                freqTable[word] = 1
        return freqTable
    
    def _score_sentences(sentences, freqTable) -> dict:
        sentenceValue = dict()
        for sentence in sentences:
            word_count_in_sentence = (len(word_tokenize(sentence)))
            for wordValue in freqTable:
                if wordValue in sentence.lower():
                    if sentence[:10] in sentenceValue:
                        sentenceValue[sentence[:10]] += freqTable[wordValue]
                    else:
                        sentenceValue[sentence[:10]] = freqTable[wordValue]
            sentenceValue[sentence[:10]] = sentenceValue[sentence[:10]] // word_count_in_sentence
        return sentenceValue

    def _find_average_score(sentenceValue) -> int:
        sumValues = 0
        for entry in sentenceValue:
            sumValues += sentenceValue[entry]
        # Average value of a sentence from original text
        average = int(sumValues / len(sentenceValue))
        return average

    def _generate_summary(sentences, sentenceValue, threshold):
        sentence_count = 0
        summary = ''
        for sentence in sentences:
            if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] > (threshold):
                summary += " " + sentence
                sentence_count += 1
        return summary

    text = MyText
    # 1 Create the word frequency table
    freq_table = _create_frequency_table(text)
    # print(freq_table)
    '''
    We already have a sentence tokenizer, so we just need
    to run the sent_tokenize() method to create the array of sentences.
    '''
    # 2 Tokenize the sentences
    sentences = sent_tokenize(text)
    # print(sentences)
    # 3 Important Algorithm: score the sentences
    sentence_scores = _score_sentences(sentences, freq_table)
    # print(sentence_scores)
    # 4 Find the threshold
    threshold = _find_average_score(sentence_scores)
    # print(threshold)
    # 5 Important Algorithm: Generate the summary
    summary = _generate_summary(sentences, sentence_scores, 1 * threshold)
    return summary

def summarize3(MyText,top_n):
    def read_article():
        filedata = MyText
        article = filedata.split(". ")
        sentences = []

        for sentence in article:
            print(sentence)
            sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
        sentences.pop() 
        
        return sentences

    def sentence_similarity(sent1, sent2, stopwords=None):
        if stopwords is None:
            stopwords = []
    
        sent1 = [w.lower() for w in sent1]
        sent2 = [w.lower() for w in sent2]
    
        all_words = list(set(sent1 + sent2))
    
        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)
    
        # build the vector for the first sentence
        for w in sent1:
            if w in stopwords:
                continue
            vector1[all_words.index(w)] += 1
    
        # build the vector for the second sentence
        for w in sent2:
            if w in stopwords:
                continue
            vector2[all_words.index(w)] += 1
    
        return 1 - cosine_distance(vector1, vector2)
    
    def build_similarity_matrix(sentences, stop_words):
        # Create an empty similarity matrix
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2: #ignore if both are same sentences
                    continue 
                similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

        return similarity_matrix


    def generate_summary():
        stop_words = stopwords.words('english')
        summarize_text = []

        # Step 1 - Read text anc split it
        sentences =  read_article()

        # Step 2 - Generate Similary Martix across sentences
        sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

        # Step 3 - Rank sentences in similarity martix
        sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
        scores = nx.pagerank(sentence_similarity_graph)

        # Step 4 - Sort the rank and pick top sentences
        ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
        print("Indexes of top ranked_sentence order are ", ranked_sentence)    

        for i in range(top_n):
            summarize_text.append(" ".join(ranked_sentence[i][1]))

        # Step 5 - Offcourse, output the summarize text
        return str(". ".join(summarize_text))

    # let's begin
    return generate_summary()

def getKeywords2(text):
    return set(keywords.keywords(text).split('\n'))

def getKeywords1(text):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    return set(rake.get_ranked_phrases())

# def getKeywords3(doc):
#     n_gram_range = (1, 1)
#     stop_words = "english"

#     # Extract candidate words/phrases
#     count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([doc])
#     candidates = count.get_feature_names()


#     model = SentenceTransformer('distilbert-base-nli-mean-tokens')
#     doc_embedding = model.encode([doc])
#     candidate_embeddings = model.encode(candidates)

#     from sklearn.metrics.pairwise import cosine_similarity

#     top_n = 5
#     distances = cosine_similarity(doc_embedding, candidate_embeddings)
#     keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]

#     return keywords

def get_cosine_sim(X,Y):
  
    # tokenization
    X_list = word_tokenize(X) 
    Y_list = word_tokenize(Y)
    
    # sw contains the list of stopwords
    sw = stopwords.words('english') 
    l1 =[];l2 =[]
    
    # remove stop words from the string
    X_set = {w for w in X_list if not w in sw} 
    Y_set = {w for w in Y_list if not w in sw}
    
    # form a set containing keywords of both strings 
    rvector = X_set.union(Y_set) 
    for w in rvector:
        if w in X_set: l1.append(1) # create a vector
        else: l1.append(0)
        if w in Y_set: l2.append(1)
        else: l2.append(0)
    c = 0
    
    # cosine formula 
    for i in range(len(rvector)):
            c+= l1[i]*l2[i]
    cosine = c / float((sum(l1)*sum(l2))**0.5)
    return cosine