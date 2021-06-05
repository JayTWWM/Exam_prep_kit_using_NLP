from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
def geter(arr):
    return arr[2]
# X = input("Enter first string: ").lower()
# Y = input("Enter second string: ").lower()
question_bank = [['What is cryptography','Cryptography is associated with the process of converting ordinary plain text into unintelligible text and vice-versa.',0],['What is steganography?','Steganography is the technique of hiding secret data within an ordinary, non-secret, file or message in order to avoid detection; the secret data is then extracted at its destination. The use of steganography can be combined with encryption as an extra step for hiding or protecting data',0],['What is your country?','India',0]]
inp = 'Y'
while inp == 'Y':
    question_bank.sort(key = geter)
    X = input(question_bank[0][0]+" : ")
    Y = question_bank[0][1]

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
    print("Correctness of answer: ", cosine)
    question_bank[0][2] = cosine
    print("Correct anser: "+question_bank[0][1])
    inp = input("Do you wish to continue?: ")