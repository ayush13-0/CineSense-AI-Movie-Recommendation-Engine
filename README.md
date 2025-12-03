## 🎬 CineSense – AI-Powered Movie Recommender System
- An intelligent movie recommendation system built using Python, Pandas, Scikit-Learn, and NLP techniques.
- CineSense analyzes movie metadata (genres, keywords, cast, crew, and overview) and recommends similar movies using Cosine Similarity.

# 🚀 Project Overview
Modern OTT platforms rely on AI-powered recommendation engines to keep users engaged, CineSense AI replicates this logic using content similarity.
CineSense AI is a content-based movie recommendation system built using NLP and machine learning, It processes movie metadata, cleans text, generates tags, transforms them into vectors, and uses cosine similarity to recommend movies closest in meaning.

The system performs:
- Metadata extraction
- Data cleaning
- Tag generation
- Text normalization & stemming
- Bag-of-Words vectorization
- Cosine similarity computation
- Top-5 movie recommendation

# 📂 Dataset Used
This project uses the TMDB 5000 Movies Dataset, containing:
- tmdb_5000_movies.csv
- tmdb_5000_credits.csv

These include rich metadata such as:
- 🎭 Genres
- 🎞️ Keywords
- 🧑‍🎤 Cast
- 🎬 Crew
- 📝 Overview
- 📊 Popularity, Ratings, Budget, Revenue

# 🏗️ Project Workflow
1️⃣ Importing Libraries
: Used for manipulation, visualization, NLP, and model building.
- import pandas as pd
- import numpy as np
- import matplotlib.pyplot as plt
- import seaborn as sns

2️⃣ Loading the Datasets
- movies = pd.read_csv("tmdb_5000_movies.csv")
- credits = pd.read_csv("tmdb_5000_credits.csv")

3️⃣ Inspecting Data
- movies.head(2)
- movies.info()

4️⃣ Merging Movie & Credits Data
- Merged using the movie title.
- movies = movies.merge(credits, on='title')

5️⃣ Feature Selection & Cleaning
Selected important fields:-
- id
- title
- overview
- genres
- keywords
- cast
- crew
Removed unnecessary columns and handled missing values.

6️⃣ Converting JSON-like Strings to Lists
Used ast.literal_eval to convert dictionary-like strings to Python objects.

7️⃣ Creating the tags Column
Combined important columns into a single unified feature text.
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# 🧠 Text Processing & NLP
The system performs:
- Lowercasing
- Removing spaces in multi-word tokens
- Stemming using PorterStemmer

- Importing Stemmer
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

- Stemming Function
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

- Applying Stemming
df['tags'] = df['tags'].apply(stem)

# 🔍 Testing the Stemmer
df['tags'][0], ps.stem('loved'), stem("in the 22nd century, a paraplegic marine is dispatched...")

# 🧮 Vectorization
- Convert textual tags into numerical vectors using Bag-of-Words.

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# 📐 Cosine Similarity Matrix
- Used to compute similarity between movies.

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)
similarity.shape

# 🎯 Recommendation Function
- Returns Top 5 Similar Movies for any movie title.

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)

# 💾 Model Saving
- To use the model later (deployment-ready):

import pickle
pickle.dump(new_df, open('movies.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))


# Distributed files
File	Description : 
- movies.pkl	Processed movie dataframe
- similarity.pkl	Cosine similarity matrix
- CineSense-AI-Recommendation-System.ipynb Full Jupyter notebook
- README.md	Documentation

# ▶️ How to Use
- Run the notebook and call:

**recommend('Avatar')**
: Output will display the top 5 recommended movies based on content similarity.

# 📈 Future Enhancements
- TF-IDF vectorization
- Word2Vec / Doc2Vec embedding
- BERT semantic embeddings
- Hybrid (Content + Collaborative filtering)
- Deployment using Flask / FastAPI
- Web UI using Streamlit or React

#🏁 Conclusion

- CineMatch AI successfully demonstrates how Natural Language Processing and machine learning can be applied to build a practical, content-based movie recommendation system.
- By converting movie metadata into feature vectors and measuring their similarity, the system provides accurate and meaningful movie suggestions.

- This project lays a strong foundation for more advanced recommendation engines and can be expanded with deep learning, hybrid methods, and full-stack deployment to create a production-grade system.

# 👨‍💻 Author

# Ayush
- 🔗 GitHub: https://github.com/ayush13-0
- 🔗 LinkedIn: https://linkedin.com/in/ayush130
