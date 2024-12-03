from flask import Flask, request, render_template
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score

app = Flask(__name__)

# Load and clean the dataset
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    # Drop rows with missing titles or genres
    df.dropna(subset=['title', 'genres'], inplace=True)

    # Normalize titles (lowercase and strip)
    df['title'] = df['title'].str.lower().str.strip()

    # Ensure genres are a list of strings
    df['genres'] = df['genres'].apply(lambda x: x.split('|') if isinstance(x, str) else [])
    return df

# Preprocess the data for machine learning
def preprocess_data(df):
    mlb = MultiLabelBinarizer()
    genre_features = mlb.fit_transform(df['genres'])
    genre_df = pd.DataFrame(genre_features, columns=mlb.classes_)
    df = pd.concat([df.reset_index(drop=True), genre_df.reset_index(drop=True)], axis=1)
    return df, genre_features

# Load and preprocess the data
df = load_and_clean_data('data/movie.csv')  # Replace with the actual file path
df, genre_features = preprocess_data(df)

# Train models
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(genre_features)

kmeans_model = KMeans(n_clusters=10, random_state=42)
df['cluster'] = kmeans_model.fit_predict(genre_features)

# Evaluate clustering with silhouette score
kmeans_silhouette = silhouette_score(genre_features, df['cluster'])
print(f"K-Means Silhouette Score: {kmeans_silhouette}")

# Data Visualization functions
def visualize_data(df):
    # Cluster Distribution Plot
    plt.figure(figsize=(12, 6))
    sns.countplot(x='cluster', data=df, palette='viridis')
    plt.title('Distribution of Movies Across Clusters')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Movies')
    plt.tight_layout()
    plt.savefig('static/cluster_distribution.png')  # Save the plot to display in Flask
    plt.close()

    # Top Genres Plot
    plt.figure(figsize=(12, 6))
    top_genres = df['genres'].explode().value_counts().head(10)
    sns.barplot(x=top_genres.values, y=top_genres.index, palette='viridis')
    plt.title('Top 10 Genres by Count')
    plt.xlabel('Count')
    plt.ylabel('Genre')
    plt.tight_layout()
    plt.savefig('static/genre_distribution.png')  # Save the plot to display in Flask
    plt.close()

# Call visualize_data to generate the charts
visualize_data(df)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    genre_features, df['cluster'], test_size=0.3, random_state=42
)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=500),
    'Support Vector Classifier': SVC(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
}

# Train and evaluate each model
metrics = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics[model_name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred, average='macro'),
        'F1 Score': f1_score(y_test, y_pred, average='macro'),
        'Confusion Matrix': confusion_matrix(y_test, y_pred).tolist(),  # Convert to list for JSON compatibility
    }

# Generate Correlation Heatmap
def correlation_heatmap(df):
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=['number'])
    
    if numeric_df.empty:
        raise ValueError("No numeric columns available for correlation.")
    
    corr = numeric_df.corr()  # Compute the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('static/correlation_heatmap.png')  # Save to static folder
    plt.close()

# Generate Genre Distribution Plot
def genre_distribution(df):
    top_genres = df['genres'].explode().value_counts().head(10)  # Top 10 genres
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_genres.values, y=top_genres.index, palette='viridis')
    plt.title('Top 10 Genres by Count')
    plt.xlabel('Count')
    plt.ylabel('Genre')
    plt.tight_layout()
    plt.savefig('static/genre_distribution.png')  # Save to static folder
    plt.close()

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/metrics')
def metrics_view():
    # Generate the heatmap and genre distribution charts
    correlation_heatmap(df) 
    genre_distribution(df)  

    # Pass metrics to the template
    return render_template(
        'metrics.html', 
        metrics=metrics,
        kmeans_silhouette=kmeans_silhouette,  # Add silhouette score to the template
        correlation_heatmap_path='static/correlation_heatmap.png',  # Path to correlation heatmap
        genre_distribution_path='static/genre_distribution.png'  # Path to genre distribution plot
    )

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form['movie_title'].strip().lower()  # Normalize input
    matches = df[df['title'].str.contains(movie_title, na=False)]  # Partial match

    if matches.empty:
        return render_template('recommend.html', 
                               movie_title=movie_title, 
                               error="Movie not found in the dataset.")

    # Use the first matched movie
    selected_movie = matches.iloc[0]['title']
    recommendations_knn = knn_recommend(selected_movie)
    recommendations_cluster = kmeans_recommend(selected_movie)
    
    return render_template('recommend.html', 
                           movie_title=selected_movie, 
                           knn=recommendations_knn, 
                           cluster=recommendations_cluster,
                           silhouette_score=kmeans_silhouette)

def knn_recommend(movie_title, n_recommendations=5):
    movie_idx = df[df['title'] == movie_title].index[0]
    distances, indices = knn_model.kneighbors([genre_features[movie_idx]], n_neighbors=n_recommendations + 1)
    recommendations = df.iloc[indices[0][1:]]['title'].values
    return recommendations

def kmeans_recommend(movie_title):
    movie_cluster = df[df['title'] == movie_title]['cluster'].values[0]
    cluster_movies = df[df['cluster'] == movie_cluster]['title'].values[:10]
    return cluster_movies

if __name__ == '__main__':
    app.run(debug=True)
