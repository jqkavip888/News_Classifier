from flask import Flask,request
import requests
import fasttext
import jieba

app = Flask(__name__)
model = fasttext.load_model('data/model.bin')

@app.route('/predict',methods=['POST'])
def index():
    uid = request.form['uid']
    sentence = request.form.get('text','')
    res = model.predict(' '.join(jieba.cut(sentence)))
    return res[0][0]

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=19999)