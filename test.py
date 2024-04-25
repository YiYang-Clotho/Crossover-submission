# Install Surprise library
# !pip install scikit-surprise


# Import necessary libraries
import pandas as pd
from surprise import SVD
from flask import Flask, request, jsonify
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split

app = Flask(__name__)

# Load the dataset
data = Dataset.load_builtin('ml-100k')

print(dir(data))

# Split the data into train and test sets
full_trainset = data.build_full_trainset()

# Define the similarity options
sim_options = {
    'name': 'cosine',
    'user_based': False  # Item-based collaborative filtering
}

# Create the KNN model
model = KNNBasic(sim_options=sim_options)

# Train the model on the training set
model.fit(full_trainset)

@app.route('/recommend', methods=['POST'])
def recommend():
    content = request.get_json()
    user_id = content['user_id']
    top_n = content['top_n']
    
    # Get the items the user hasn't rated
    items_to_recommend = [item_id for item_id in full_trainset.all_items() if item_id not in full_trainset.ur[user_id]]
    
    # Predict ratings for the items
    item_ratings = {}
    for item_id in items_to_recommend:
        rating = model.predict(user_id, item_id).est
        item_ratings[item_id] = rating
    
    # Get the top-N recommendations
    top_n_recommendations = sorted(item_ratings.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Format the recommendations
    recommendations = [{'item_id': item_id, 'rating': rating} for item_id, rating in top_n_recommendations]
    
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)