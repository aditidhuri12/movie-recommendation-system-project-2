import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Sample movie dataset (you can expand this or load from a CSV)
data = {
    'title': ['The Matrix', 'John Wick', 'Inception', 'Interstellar', 'The Dark Knight', 'Avengers', 'Iron Man'],
    'genres': [
        'Action Sci-Fi',
        'Action Thriller',
        'Action Sci-Fi Thriller',
        'Adventure Drama Sci-Fi',
        'Action Crime Drama',
        'Action Adventure Sci-Fi',
        'Action Sci-Fi'
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# TF-IDF Vectorizer on genres
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['genres'])

# Compute cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get recommendations
def recommend(title, cosine_sim=cosine_sim):
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]  # Top 3 recommendations
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]

# Example: Recommend movies similar to "The Matrix"
print("Recommended for 'The Matrix':")
print(recommend('The Matrix'))
