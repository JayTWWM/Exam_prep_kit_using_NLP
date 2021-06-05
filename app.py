from flask import Flask,request,render_template
from stt import getKeywords1
from nltk.corpus import wordnet
import json
import scipy
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')

  
app = Flask(__name__)
app.config["DEBUG"] = True
qa = {}
f = open(r"D:\Machine Learning\NLP\question.json")
data = json.load(f)
arr = []
for a in data:
    arr.append([a,0])
    qa[a] = data[a]
f.close()

arr.sort(key=lambda x:x[1])
que = arr[0][0]

@app.route('/')
def home():
    qa = {}
    f = open(r"D:\Machine Learning\NLP\question.json")
    data = json.load(f)
    arr = []
    for a in data:
        arr.append([a,0])
        qa[a] = data[a]
    f.close()
    arr.sort(key=lambda x:x[1])
    que = arr[0][0]
    return render_template('index.html',que=que)

@app.route('/predict', methods=['POST'])
def predict():
    global que
    answer = request.form['body']
    marks = score1(answer,que)
    arr.sort(key=lambda x:x[1])
    que = arr[0][0]
    return render_template('index.html',que=que,status=marks)

def score(ans,que):
    ogAns = qa[que]
    ans = getKeywords1(ans)
    ogAns = getKeywords1(ogAns)
    if not ans:
        for i in range(len(arr)):
            if arr[i][0]==que:
                arr[i][1] = 0
        return 0
    correct = 0
    for a in ogAns:
        f = False 
        if a in ans:
            correct+=1
            continue
        try:
            synSet = wordnet.synsets(a)
            if synSet:
                for syn in synSet:
                    if not f:
                        for l in syn.lemmas():
                            if l.name() in ans:
                                correct+=1 
                                f = True
                                break
        except:
            continue
    marks = round(correct/len(ogAns)*100,2)
    for i in range(len(arr)):
        if arr[i][0]==que:
            arr[i][1] = marks
    return marks

def score1(ans,que):
    # Convert the corpus into a list of headlines
    corpus=[i for i in ans.split('\n')if i != ''and len(i.split(' '))>=4]
    # Get a vector for each headline (sentence) in the corpus
    corpus_embeddings = model.encode(corpus)
    # Define search queries and embed them to vectors as well
    answer = qa[que]
    queries = answer.split('.')[:-1]
    query_embeddings = model.encode(queries)
    # For each search term return 5 closest sentences
    correct = 0 
    closest_n = 1
    for query, query_embedding in zip(queries, query_embeddings):
        distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        print("\n\n======================\n\n")
        print("Query:", query)

        for idx, distance in results[0:closest_n]:
            print(corpus[idx].strip(), "(Score: %.4f)" % (1-distance))
            if (1-distance)>=0.8:
                correct+=1
            elif (1-distance)>=0.5:
                correct+=0.5
    for i in range(len(arr)):
        if arr[i][0]==que:
            arr[i][1] = correct
    return correct
app.run()