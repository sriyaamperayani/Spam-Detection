{
"cells": [
{
"cell_type": "markdown",
"metadata": {},
"source": [
"# spam detection with natural language processing "
]
},
{
"cell_type": "code",
"execution_count": 66,
"metadata": {},
"outputs": [],
"source": [
"import numpy as np   \n",
"import pandas as pd  "
]
},
{
"cell_type": "code",
"execution_count": 1,
"metadata": {},
"outputs": [
{
"name": "stdout",
"output_type": "stream",
"text": [
"Requirement already satisfied: nltk in c:\\users\\jahnavi\\anaconda3\\lib\\site-packages (3.4)\n",
"Requirement already satisfied: six in c:\\users\\jahnavi\\anaconda3\\lib\\site-packages (from nltk) (1.12.0)\n",
"Requirement already satisfied: singledispatch in c:\\users\\jahnavi\\anaconda3\\lib\\site-packages (from nltk) (3.4.0.3)\n",
"Note: you may need to restart the kernel to use updated packages.\n"
]
}
],
"source": [
"pip install nltk"
]
},
{
"cell_type": "code",
"execution_count": null,
"metadata": {},
"outputs": [
{
"name": "stdout",
"output_type": "stream",
"text": [
"showing info https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml\n"
]
}
],
"source": [
"import nltk\n",
"nltk.download() "
]
},
{
"cell_type": "markdown",
"metadata": {},
"source": [
"# 1) LOADING DATA SET "
]
},
{
"cell_type": "code",
"execution_count": 67,
"metadata": {},
"outputs": [
{
"data": {
"text/html": [
"<div>\n",
"<style scoped>\n",
"    .dataframe tbody tr th:only-of-type {\n",
"        vertical-align: middle;\n",
"    }\n",
"\n",
"    .dataframe tbody tr th {\n",
"        vertical-align: top;\n",
"    }\n",
"\n",
"    .dataframe thead th {\n",
"        text-align: right;\n",
"    }\n",
"</style>\n",
"<table border=\"1\" class=\"dataframe\">\n",
"  <thead>\n",
"    <tr style=\"text-align: right;\">\n",
"      <th></th>\n",
"      <th>type</th>\n",
"      <th>text</th>\n",
"    </tr>\n",
"  </thead>\n",
"  <tbody>\n",
"    <tr>\n",
"      <th>0</th>\n",
"      <td>ham</td>\n",
"      <td>Go until jurong point, crazy.. Available only ...</td>\n",
"    </tr>\n",
"    <tr>\n",
"      <th>1</th>\n",
"      <td>ham</td>\n",
"      <td>Ok lar... Joking wif u oni...</td>\n",
"    </tr>\n",
"    <tr>\n",
"      <th>2</th>\n",
"      <td>spam</td>\n",
"      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
"    </tr>\n",
"    <tr>\n",
"      <th>3</th>\n",
"      <td>ham</td>\n",
"      <td>U dun say so early hor... U c already then say...</td>\n",
"    </tr>\n",
"    <tr>\n",
"      <th>4</th>\n",
"      <td>ham</td>\n",
"      <td>Nah I don't think he goes to usf, he lives aro...</td>\n",
"    </tr>\n",
"    <tr>\n",
"      <th>5</th>\n",
"      <td>spam</td>\n",
"      <td>FreeMsg Hey there darling it's been 3 week's n...</td>\n",
"    </tr>\n",
"    <tr>\n",
"      <th>6</th>\n",
"      <td>ham</td>\n",
"      <td>Even my brother is not like to speak with me. ...</td>\n",
"    </tr>\n",
"    <tr>\n",
"      <th>7</th>\n",
"      <td>ham</td>\n",
"      <td>As per your request 'Melle Melle (Oru Minnamin...</td>\n",
"    </tr>\n",
"    <tr>\n",
"      <th>8</th>\n",
"      <td>spam</td>\n",
"      <td>WINNER!! As a valued network customer you have...</td>\n",
"    </tr>\n",
"    <tr>\n",
"      <th>9</th>\n",
"      <td>spam</td>\n",
"      <td>Had your mobile 11 months or more? U R entitle...</td>\n",
"    </tr>\n",
"  </tbody>\n",
"</table>\n",
"</div>"
],
"text/plain": [
"   type                                               text\n",
"0   ham  Go until jurong point, crazy.. Available only ...\n",
"1   ham                      Ok lar... Joking wif u oni...\n",
"2  spam  Free entry in 2 a wkly comp to win FA Cup fina...\n",
"3   ham  U dun say so early hor... U c already then say...\n",
"4   ham  Nah I don't think he goes to usf, he lives aro...\n",
"5  spam  FreeMsg Hey there darling it's been 3 week's n...\n",
"6   ham  Even my brother is not like to speak with me. ...\n",
"7   ham  As per your request 'Melle Melle (Oru Minnamin...\n",
"8  spam  WINNER!! As a valued network customer you have...\n",
"9  spam  Had your mobile 11 months or more? U R entitle..."
]
},
"execution_count": 67,
"metadata": {},
"output_type": "execute_result"
}
],
"source": [
"dt = pd.read_csv(\"spam.csv\")  \n",
"dt.head(10)"
]
},
{
"cell_type": "code",
"execution_count": 68,
"metadata": {},
"outputs": [
{
"data": {
"text/html": [
"<div>\n",
"<style scoped>\n",
"    .dataframe tbody tr th:only-of-type {\n",
"        vertical-align: middle;\n",
"    }\n",
"\n",
"    .dataframe tbody tr th {\n",
"        vertical-align: top;\n",
"    }\n",
"\n",
"    .dataframe thead th {\n",
"        text-align: right;\n",
"    }\n",
"</style>\n",
"<table border=\"1\" class=\"dataframe\">\n",
"  <thead>\n",
"    <tr style=\"text-align: right;\">\n",
"      <th></th>\n",
"      <th>type</th>\n",
"      <th>text</th>\n",
"      <th>spam</th>\n",
"    </tr>\n",
"  </thead>\n",
"  <tbody>\n",
"    <tr>\n",
"      <th>0</th>\n",
"      <td>ham</td>\n",
"      <td>Go until jurong point, crazy.. Available only ...</td>\n",
"      <td>0</td>\n",
"    </tr>\n",
"    <tr>\n",
"      <th>1</th>\n",
"      <td>ham</td>\n",
"      <td>Ok lar... Joking wif u oni...</td>\n",
"      <td>0</td>\n",
"    </tr>\n",
"    <tr>\n",
"      <th>2</th>\n",
"      <td>spam</td>\n",
"      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
"      <td>1</td>\n",
"    </tr>\n",
"    <tr>\n",
"      <th>3</th>\n",
"      <td>ham</td>\n",
"      <td>U dun say so early hor... U c already then say...</td>\n",
"      <td>0</td>\n",
"    </tr>\n",
"    <tr>\n",
"      <th>4</th>\n",
"      <td>ham</td>\n",
"      <td>Nah I don't think he goes to usf, he lives aro...</td>\n",
"      <td>0</td>\n",
"    </tr>\n",
"  </tbody>\n",
"</table>\n",
"</div>"
],
"text/plain": [
"   type                                               text  spam\n",
"0   ham  Go until jurong point, crazy.. Available only ...     0\n",
"1   ham                      Ok lar... Joking wif u oni...     0\n",
"2  spam  Free entry in 2 a wkly comp to win FA Cup fina...     1\n",
"3   ham  U dun say so early hor... U c already then say...     0\n",
"4   ham  Nah I don't think he goes to usf, he lives aro...     0"
]
},
"execution_count": 68,
"metadata": {},
"output_type": "execute_result"
}
],
"source": [
"dt['spam'] = dt['type'].map( {'spam': 1, 'ham': 0} ).astype(int)\n",
"dt.head(5)"
]
},
{
"cell_type": "code",
"execution_count": 69,
"metadata": {},
"outputs": [
{
"name": "stdout",
"output_type": "stream",
"text": [
"COLUMS IN THE GIVEN DATA:\n",
"type\n",
"text\n",
"spam\n"
]
}
],
"source": [
"print(\"COLUMS IN THE GIVEN DATA:\")\n",
"for col in dt.columns: \n",
"    print(col) "
]
},
{
"cell_type": "code",
"execution_count": 70,
"metadata": {},
"outputs": [
{
"name": "stdout",
"output_type": "stream",
"text": [
"NO OF ROWS IN REVIEW COLUMN: 5574\n",
"NO OF ROWS IN liked COLUMN: 5574\n"
]
}
],
"source": [
"t=len(dt['type'])\n",
"print(\"NO OF ROWS IN REVIEW COLUMN:\",t)\n",
"t=len(dt['text'])\n",
"print(\"NO OF ROWS IN liked COLUMN:\",t)"
]
},
{
"cell_type": "markdown",
"metadata": {},
"source": [
"# 2)Tokenization"
]
},
{
"cell_type": "code",
"execution_count": 71,
"metadata": {},
"outputs": [
{
"data": {
"text/plain": [
"'Ok lar... Joking wif u oni...'"
]
},
"execution_count": 71,
"metadata": {},
"output_type": "execute_result"
}
],
"source": [
"dt['text'][1]#before   "
]
},
{
"cell_type": "code",
"execution_count": 72,
"metadata": {},
"outputs": [],
"source": [
"def tokenizer(text):\n",
"    return text.split()"
]
},
{
"cell_type": "code",
"execution_count": 73,
"metadata": {},
"outputs": [],
"source": [
"dt['text']=dt['text'].apply(tokenizer)"
]
},
{
"cell_type": "code",
"execution_count": 74,
"metadata": {},
"outputs": [
{
"data": {
"text/plain": [
"['Ok', 'lar...', 'Joking', 'wif', 'u', 'oni...']"
]
},
"execution_count": 74,
"metadata": {},
"output_type": "execute_result"
}
],
"source": [
"dt['text'][1]#after "
]
},
{
"cell_type": "markdown",
"metadata": {},
"source": [
"# 3) STEMMING "
]
},
{
"cell_type": "code",
"execution_count": 75,
"metadata": {},
"outputs": [
{
"data": {
"text/plain": [
"['Ok', 'lar...', 'Joking', 'wif', 'u', 'oni...']"
]
},
"execution_count": 75,
"metadata": {},
"output_type": "execute_result"
}
],
"source": [
"dt['text'][1]#before "
]
},
{
"cell_type": "code",
"execution_count": 76,
"metadata": {},
"outputs": [],
"source": [
"from nltk.stem.snowball import SnowballStemmer\n",
"porter = SnowballStemmer(\"english\", ignore_stopwords=False)"
]
},
{
"cell_type": "code",
"execution_count": 77,
"metadata": {},
"outputs": [],
"source": [
"def stem_it(text):\n",
"    return [porter.stem(word) for word in text]"
]
},
{
"cell_type": "code",
"execution_count": 78,
"metadata": {},
"outputs": [],
"source": [
"dt['text']=dt['text'].apply(stem_it)"
]
},
{
"cell_type": "code",
"execution_count": 79,
"metadata": {},
"outputs": [
{
"data": {
"text/plain": [
"['ok', 'lar...', 'joke', 'wif', 'u', 'oni...']"
]
},
"execution_count": 79,
"metadata": {},
"output_type": "execute_result"
}
],
"source": [
"dt['text'][1] #after stemming "
]
},
{
"cell_type": "markdown",
"metadata": {},
"source": [
"# 4) LEMMITIZATION "
]
},
{
"cell_type": "code",
"execution_count": 80,
"metadata": {},
"outputs": [
{
"data": {
"text/plain": [
"['yup',\n",
" 'i',\n",
" 'thk',\n",
" 'cine',\n",
" 'is',\n",
" 'better',\n",
" 'cos',\n",
" 'no',\n",
" 'need',\n",
" '2',\n",
" 'go',\n",
" 'down',\n",
" '2',\n",
" 'plaza',\n",
" 'mah.']"
]
},
"execution_count": 80,
"metadata": {},
"output_type": "execute_result"
}
],
"source": [
"dt['text'][152] #before"
]
},
{
"cell_type": "code",
"execution_count": 81,
"metadata": {},
"outputs": [],
"source": [
"from nltk.stem import WordNetLemmatizer \n",
"lemmatizer = WordNetLemmatizer() "
]
},
{
"cell_type": "code",
"execution_count": 82,
"metadata": {},
"outputs": [],
"source": [
"def lemmit_it(text):\n",
"    return [lemmatizer.lemmatize(word, pos =\"a\") for word in text]"
]
},
{
"cell_type": "code",
"execution_count": 83,
"metadata": {},
"outputs": [],
"source": [
"dt['text']=dt['text'].apply(lemmit_it)"
]
},
{
"cell_type": "code",
"execution_count": 84,
"metadata": {},
"outputs": [
{
"data": {
"text/plain": [
"['yup',\n",
" 'i',\n",
" 'thk',\n",
" 'cine',\n",
" 'is',\n",
" 'good',\n",
" 'cos',\n",
" 'no',\n",
" 'need',\n",
" '2',\n",
" 'go',\n",
" 'down',\n",
" '2',\n",
" 'plaza',\n",
" 'mah.']"
]
},
"execution_count": 84,
"metadata": {},
"output_type": "execute_result"
}
],
"source": [
"dt['text'][152] #before"
]
},
{
"cell_type": "markdown",
"metadata": {},
"source": [
"# 5)STOPWORD REMOVAL"
]
},
{
"cell_type": "code",
"execution_count": 85,
"metadata": {},
"outputs": [
{
"data": {
"text/plain": [
"['tired.', 'i', \"haven't\", 'slept', 'well', 'the', 'past', 'few', 'nights.']"
]
},
"execution_count": 85,
"metadata": {},
"output_type": "execute_result"
}
],
"source": [
"dt['text'][217] #before "
]
},
{
"cell_type": "code",
"execution_count": 86,
"metadata": {},
"outputs": [],
"source": [
"#from nltk.corpus import stopwords\n",
"#print(stopwords.words('english'))\n",
"#set(['a','t','d','y'])"
]
},
{
"cell_type": "code",
"execution_count": 87,
"metadata": {},
"outputs": [],
"source": [
"from nltk.corpus import stopwords\n",
"stop_words = stopwords.words('english')"
]
},
{
"cell_type": "code",
"execution_count": 88,
"metadata": {},
"outputs": [],
"source": [
"def stop_it(text):\n",
"    review = [word for word in text if not word in stop_words ] \n",
"    return review"
]
},
{
"cell_type": "code",
"execution_count": 89,
"metadata": {},
"outputs": [],
"source": [
"dt['text']=dt['text'].apply(stop_it)"
]
},
{
"cell_type": "code",
"execution_count": 90,
"metadata": {},
"outputs": [
{
"data": {
"text/plain": [
"['tired.', 'slept', 'well', 'past', 'nights.']"
]
},
"execution_count": 90,
"metadata": {},
"output_type": "execute_result"
}
],
"source": [
"dt['text'][217] #after "
]
},
{
"cell_type": "code",
"execution_count": 91,
"metadata": {},
"outputs": [
{
"data": {
"text/html": [
"<div>\n",
"<style scoped>\n",
"    .dataframe tbody tr th:only-of-type {\n",
"        vertical-align: middle;\n",
"    }\n",
"\n",
"    .dataframe tbody tr th {\n",
"        vertical-align: top;\n",
"    }\n",
"\n",
"    .dataframe thead th {\n",
"        text-align: right;\n",
"    }\n",
"</style>\n",
"<table border=\"1\" class=\"dataframe\">\n",
"  <thead>\n",
"    <tr style=\"text-align: right;\">\n",
"      <th></th>\n",
"      <th>type</th>\n",
"      <th>text</th>\n",
"      <th>spam</th>\n",
"    </tr>\n",
"  </thead>\n",
"  <tbody>\n",
"    <tr>\n",
"      <th>0</th>\n",
"      <td>ham</td>\n",
"      <td>[go, jurong, point,, crazy.., avail, onli, bug...</td>\n",
"      <td>0</td>\n",
"    </tr>\n",
"    <tr>\n",
"      <th>1</th>\n",
"      <td>ham</td>\n",
"      <td>[ok, lar..., joke, wif, u, oni...]</td>\n",
"      <td>0</td>\n",
"    </tr>\n",
"    <tr>\n",
"      <th>2</th>\n",
"      <td>spam</td>\n",
"      <td>[free, entri, 2, wkli, comp, win, fa, cup, fin...</td>\n",
"      <td>1</td>\n",
"    </tr>\n",
"    <tr>\n",
"      <th>3</th>\n",
"      <td>ham</td>\n",
"      <td>[u, dun, say, earli, hor..., u, c, alreadi, sa...</td>\n",
"      <td>0</td>\n",
"    </tr>\n",
"    <tr>\n",
"      <th>4</th>\n",
"      <td>ham</td>\n",
"      <td>[nah, think, goe, usf,, live, around, though]</td>\n",
"      <td>0</td>\n",
"    </tr>\n",
"    <tr>\n",
"      <th>5</th>\n",
"      <td>spam</td>\n",
"      <td>[freemsg, hey, darl, 3, week, word, back!, i'd...</td>\n",
"      <td>1</td>\n",
"    </tr>\n",
"    <tr>\n",
"      <th>6</th>\n",
"      <td>ham</td>\n",
"      <td>[even, brother, like, speak, me., treat, like,...</td>\n",
"      <td>0</td>\n",
"    </tr>\n",
"    <tr>\n",
"      <th>7</th>\n",
"      <td>ham</td>\n",
"      <td>[per, request, mell, mell, (oru, minnaminungin...</td>\n",
"      <td>0</td>\n",
"    </tr>\n",
"    <tr>\n",
"      <th>8</th>\n",
"      <td>spam</td>\n",
"      <td>[winner!!, valu, network, custom, select, rece...</td>\n",
"      <td>1</td>\n",
"    </tr>\n",
"    <tr>\n",
"      <th>9</th>\n",
"      <td>spam</td>\n",
"      <td>[mobil, 11, month, more?, u, r, entitl, updat,...</td>\n",
"      <td>1</td>\n",
"    </tr>\n",
"  </tbody>\n",
"</table>\n",
"</div>"
],
"text/plain": [
"   type                                               text  spam\n",
"0   ham  [go, jurong, point,, crazy.., avail, onli, bug...     0\n",
"1   ham                 [ok, lar..., joke, wif, u, oni...]     0\n",
"2  spam  [free, entri, 2, wkli, comp, win, fa, cup, fin...     1\n",
"3   ham  [u, dun, say, earli, hor..., u, c, alreadi, sa...     0\n",
"4   ham      [nah, think, goe, usf,, live, around, though]     0\n",
"5  spam  [freemsg, hey, darl, 3, week, word, back!, i'd...     1\n",
"6   ham  [even, brother, like, speak, me., treat, like,...     0\n",
"7   ham  [per, request, mell, mell, (oru, minnaminungin...     0\n",
"8  spam  [winner!!, valu, network, custom, select, rece...     1\n",
"9  spam  [mobil, 11, month, more?, u, r, entitl, updat,...     1"
]
},
"execution_count": 91,
"metadata": {},
"output_type": "execute_result"
}
],
"source": [
"dt.head(10)"
]
},
{
"cell_type": "code",
"execution_count": 92,
"metadata": {},
"outputs": [],
"source": [
"dt['text']=dt['text'].apply(' '.join)"
]
},
{
"cell_type": "code",
"execution_count": 93,
"metadata": {},
"outputs": [
{
"data": {
"text/html": [
"<div>\n",
"<style scoped>\n",
"    .dataframe tbody tr th:only-of-type {\n",
"        vertical-align: middle;\n",
"    }\n",
"\n",
"    .dataframe tbody tr th {\n",
"        vertical-align: top;\n",
"    }\n",
"\n",
"    .dataframe thead th {\n",
"        text-align: right;\n",
"    }\n",
"</style>\n",
"<table border=\"1\" class=\"dataframe\">\n",
"  <thead>\n",
"    <tr style=\"text-align: right;\">\n",
"      <th></th>\n",
"      <th>type</th>\n",
"      <th>text</th>\n",
"      <th>spam</th>\n",
"    </tr>\n",
"  </thead>\n",
"  <tbody>\n",
"    <tr>\n",
"      <th>0</th>\n",
"      <td>ham</td>\n",
"      <td>go jurong point, crazy.. avail onli bugi n gre...</td>\n",
"      <td>0</td>\n",
"    </tr>\n",
"    <tr>\n",
"      <th>1</th>\n",
"      <td>ham</td>\n",
"      <td>ok lar... joke wif u oni...</td>\n",
"      <td>0</td>\n",
"    </tr>\n",
"    <tr>\n",
"      <th>2</th>\n",
"      <td>spam</td>\n",
"      <td>free entri 2 wkli comp win fa cup final tkts 2...</td>\n",
"      <td>1</td>\n",
"    </tr>\n",
"    <tr>\n",
"      <th>3</th>\n",
"      <td>ham</td>\n",
"      <td>u dun say earli hor... u c alreadi say...</td>\n",
"      <td>0</td>\n",
"    </tr>\n",
"    <tr>\n",
"      <th>4</th>\n",
"      <td>ham</td>\n",
"      <td>nah think goe usf, live around though</td>\n",
"      <td>0</td>\n",
"    </tr>\n",
"  </tbody>\n",
"</table>\n",
"</div>"
],
"text/plain": [
"   type                                               text  spam\n",
"0   ham  go jurong point, crazy.. avail onli bugi n gre...     0\n",
"1   ham                        ok lar... joke wif u oni...     0\n",
"2  spam  free entri 2 wkli comp win fa cup final tkts 2...     1\n",
"3   ham          u dun say earli hor... u c alreadi say...     0\n",
"4   ham              nah think goe usf, live around though     0"
]
},
"execution_count": 93,
"metadata": {},
"output_type": "execute_result"
}
],
"source": [
"dt.head()"
]
},
{
"cell_type": "markdown",
"metadata": {},
"source": [
"#  6) Transform Text Data into TDF /TF-IDF Vectors   "
]
},
{
"cell_type": "code",
"execution_count": 94,
"metadata": {},
"outputs": [],
"source": [
"from sklearn.feature_extraction.text import TfidfVectorizer\n",
"tfidf=TfidfVectorizer()\n",
"y=dt.spam.values\n",
"x=tfidf.fit_transform(dt['text'])"
]
},
{
"cell_type": "code",
"execution_count": 95,
"metadata": {},
"outputs": [],
"source": [
"from sklearn.model_selection import train_test_split\n",
"x_train,x_text,y_train,y_text=train_test_split(x,y,random_state=1,test_size=0.2,shuffle=False)"
]
},
{
"cell_type": "markdown",
"metadata": {},
"source": [
"#  7) Classification using Logistic Regression"
]
},
{
"cell_type": "code",
"execution_count": 96,
"metadata": {},
"outputs": [
{
"name": "stdout",
"output_type": "stream",
"text": [
"accuracy: 96.05381165919282\n"
]
},
{
"name": "stderr",
"output_type": "stream",
"text": [
"C:\\Users\\JAHNAVI\\Anaconda3\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
"  FutureWarning)\n"
]
}
],
"source": [
"from sklearn.linear_model import LogisticRegression\n",
"clf=LogisticRegression()\n",
"clf.fit(x_train,y_train)\n",
"y_pred=clf.predict(x_text)\n",
"from sklearn.metrics import accuracy_score\n",
"acc_log = accuracy_score(y_pred, y_text)*100\n",
"print(\"accuracy:\",acc_log )"
]
},
{
"cell_type": "markdown",
"metadata": {},
"source": [
"#   8) Classification using LinearSVC Accuracy "
]
},
{
"cell_type": "code",
"execution_count": 97,
"metadata": {},
"outputs": [
{
"name": "stdout",
"output_type": "stream",
"text": [
"accuracy: 97.66816143497758\n"
]
}
],
"source": [
"from sklearn.svm import LinearSVC\n",
"\n",
"linear_svc = LinearSVC(random_state=0)\n",
"linear_svc.fit(x_train, y_train)\n",
"y_pred = linear_svc.predict(x_text)\n",
"acc_linear_svc =accuracy_score(y_pred, y_text) * 100\n",
"print(\"accuracy:\",acc_linear_svc)"
]
},
{
"cell_type": "markdown",
"metadata": {},
"source": [
"###                        ----------XXXX---------"
]
}
],
"metadata": {
"kernelspec": {
"display_name": "Python 3",
"language": "python",
"name": "python3"
},
"language_info": {
"codemirror_mode": {
"name": "ipython",
"version": 3
},
"file_extension": ".py",
"mimetype": "text/x-python",
"name": "python",
"nbconvert_exporter": "python",
"pygments_lexer": "ipython3",
"version": "3.7.3"
}
},
"nbformat": 4,
"nbformat_minor": 2
}
