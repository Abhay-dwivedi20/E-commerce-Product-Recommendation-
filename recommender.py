# recommender.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and group data
df = pd.read_csv("cleaned_amazon_data.csv")
df['img_link'] = df['img_link'].astype(str).str.strip("'").str.strip('"')

df['img_link'] = df['img_link'].fillna('https://via.placeholder.com/150')


grouped_df = df.groupby('product_id').agg({
    'product_name': 'first',
    'img_link': 'first',
    'about_product': lambda x: ' '.join(x.dropna().astype(str)),
    'category': 'first',
    'review_content': lambda x: ' '.join(x.dropna().astype(str))
}).reset_index()

# Combine features
grouped_df['combined_features'] = grouped_df['about_product'] + ' ' + \
                                  grouped_df['category'] + ' ' + \
                                  grouped_df['review_content']

# TF-IDF & similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(grouped_df['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
product_indices = pd.Series(grouped_df.index, index=grouped_df['product_id'])

# Recommend function with image links
def recommend_similar_products(product_id, top_n=20):
    if product_id not in product_indices:
        return []

    idx = int(product_indices[product_id])  # Index of input product
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    top_indexes = [i[0] for i in sim_scores]

    output = []
    for idx in top_indexes:
        product = df.iloc[idx]
        img = product['img_link']

        # Fallback if image is missing or invalid
        if not isinstance(img, str) or not img.startswith('http'):
            img = 'https://via.placeholder.com/300x300.png?text=No+Image'

        output.append({
            'product_id': product['product_id'],
            'product_name': product['product_name'],
            'img_link': img
        })

    return output  # ✅ Return the safe, cleaned list
