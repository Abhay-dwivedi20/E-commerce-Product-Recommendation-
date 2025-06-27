# app.py

from flask import Flask, render_template, request
from recommender import recommend_similar_products

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    product_id = ""
    if request.method == 'POST':
        product_id = request.form['product_id']
        recommendations = recommend_similar_products(product_id) 
         # ✅ Fix image links that are missing or broken
        for item in recommendations:
            if not item.get('img_link') or not str(item['img_link']).startswith('http'):
                item['img_link'] = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRcCBHgbS23kyBw2r8Pquu19UtKZnrZmFUx1g&s'
    return render_template('index.html', recommendations=recommendations, product_id=product_id)

if __name__ == '__main__':
    app.run(debug=True)
