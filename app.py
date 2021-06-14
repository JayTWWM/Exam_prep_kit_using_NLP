from flask import Flask,request,render_template
from stt import getKeywords1, punctuate, textSpeech
from nltk.corpus import wordnet
import json
import scipy
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')
  
app = Flask(__name__)
app.config["DEBUG"] = False
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
    global marks
    global que
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
    return render_template('index.html',que=que,ans=qa[que][0])

@app.route('/predict')
def predict():
    global que
    arr.sort(key=lambda x:x[1])
    que = arr[0][0]
    return render_template('index.html',que=que,ans=qa[que][0])

@app.route('/result', methods=['POST'])
def result():
    answer = punctuate(request.form['body'])
    marks,eval = score1(answer,que)
    return render_template('result.html',status=marks,ans=qa[que][0],exp=qa[que][1],eval=eval)

def score(ans,que):
    ogAns = qa[que][0]
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
    mp = {}
    corpus=[i for i in ans.split('\n')if i != ''and len(i.split(' '))>=4]
    corpus_embeddings = model.encode(corpus)
    answer = qa[que][0]
    queries = answer.split('.')[:-1]
    query_embeddings = model.encode(queries)
    correct = 0 
    closest_n = 1
    for query, query_embedding in zip(queries, query_embeddings):
        mp[query] = "Wrong/Incorrect/Missed"
        distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]
        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])
        print("\n\n======================\n\n")
        print("Query:", query)
        for idx, distance in results[0:closest_n]:
            print(corpus[idx].strip(), "(Score: %.4f)" % (1-distance))
            if (1-distance)>=0.8:
                correct+=1
                mp[query] = "Correct"
            elif (1-distance)>=0.5:
                correct+=0.5
                mp[query] = "Insufficiently stated"
    for i in range(len(arr)):
        if arr[i][0]==que:
            arr[i][1] = correct
    return (100*correct/len(queries),mp)
app.run()