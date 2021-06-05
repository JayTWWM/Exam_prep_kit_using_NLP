from stt import textSpeech,punctuate,summarize1,getKeywords2,get_cosine_sim,getKeywords1
from nltk.corpus import wordnet
question_bank = [['What is cryptography',"Supervised learning is the machine learning task of learning a function that maps an input to an output based on example input-output pairs. It infers a function from labeled training data consisting of a set of training examples. In supervised learning, each example is a pair consisting of an input object (typically a vector) and a desired output value (also called the supervisory signal). A supervised learning algorithm analyzes the training data and produces an inferred function, which can be used for mapping new examples. An optimal scenario will allow for the algorithm to correctly determine the class labels for unseen instances. This requires the learning algorithm to generalize from the training data to unseen situations in a 'reasonable' way (see inductive bias).",0],['What is steganography?','Steganography is the technique of hiding secret data within an ordinary, non-secret, file or message in order to avoid detection; the secret data is then extracted at its destination. The use of steganography can be combined with encryption as an extra step for hiding or protecting data',0],['What is your country?','India',0]]
inp = 'Y'
while inp == 'Y':
    question_bank.sort(key = lambda x: x )
    print(question_bank[0][0]+" : ")
    # X = textSpeech(False)
    # print(X)
    # X = punctuate(X)
    # print(X)
    # X = summarize1(X)
    X = "It infers a function from labeled training data consisting of a set of training examples. In supervised learning, each example is a pair consisting of an input object (typically a vector) and a desired output value (also called the supervisory signal). A supervised learning algorithm analyzes the training data and produces an inferred function, which can be used for mapping new examples. An optimal scenario will allow for the algorithm to correctly determine the class labels for unseen instances. This requires the learning algorithm to generalize from the training data to unseen situations in a 'reasonable' way (see inductive bias)."
    correct = 0 
    Y = question_bank[0][1]
    X = getKeywords1(X)
    print(X)
    Y = getKeywords1(Y)
    print(Y)
    for a in Y:
        f = False 
        if a in X:
            correct+=1
            continue
        for syn in wordnet.synsets(a):
            if not f:
                for l in syn.lemmas():
                    if l.name() in X:
                        correct+=1 
                        f = True
                        break 
    print(f"Marks %: {round(correct/len(Y)*100,2)}")
    # cosine = get_cosine_sim(X,Y)    
    # print("Correctness of answer: ", cosine)
    # question_bank[0]  = cosine
    # print("Correct anser: "+question_bank[0][1])
    inp = input("Do you wish to continue?: ")