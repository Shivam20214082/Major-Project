import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from preprocess import preprocess_data
from feature_engineering import calculate_anomaly_durations
import os

def apply_kmeans_clustering(df, features_to_normalize, sensor_columns, k=30):
    print("Applying KMeans clustering...")
    clustering_features = (
        [f'{col}_normalized' for col in features_to_normalize['trend_features']] +
        [f'{col}_normalized' for col in features_to_normalize['severity']] +
        ['anomaly_duration_normalized', 'correlation_behavior_normalized']
    )

    df = df.dropna(subset=clustering_features).copy()

    kmeans = KMeans(n_clusters=k, random_state=0)
    df.loc[:, 'cluster'] = kmeans.fit_predict(df[clustering_features])

    mapping = {}
    for c in range(k):
        cluster_data = df[df['cluster'] == c]['anomaly_class']
        fault_ratio = (cluster_data == 'Fault').mean()
        mapping[c] = 'Fault' if fault_ratio > 0.6 else 'Environmental Changes'

    df.loc[:, 'cluster_mapped_class'] = df['cluster'].map(mapping)
    df.loc[:, 'final_class'] = df['cluster_mapped_class']
    print("KMeans clustering applied successfully.")
    return df

def classify_with_dynamic_thresholds(row, high_score_threshold, high_corr_threshold, mid_score_threshold, mid_corr_threshold):
    if row['combined_anomaly'] == 1:
        return "Normal"
    if row['combined_anomaly'] == -1:
        if row['weighted_score'] > high_score_threshold and row['correlation_behavior'] > high_corr_threshold:
            return "Environmental Changes"
        else:
            return "Fault"
    return "Normal"

def run_classification(df):
    print("Starting classification process...")

    sensor_columns = ['temperature', 'humidity', 'mq2_analog', 'mq9_analog', 'sound_analog', 'pm25_density', 'pm10_density']
    
    # Feature engineering: calculating differences and rolling averages
    print("Calculating feature engineering metrics...")
    for col in sensor_columns:
        df.loc[:, f'{col}_diff'] = df[col].diff()
        df.loc[:, f'{col}_rolling_avg'] = df[col].rolling(window=5).mean()

    for col in sensor_columns:
        normal_mean = df[df['combined_anomaly'] == 1][col].mean()
        normal_std = df[df['combined_anomaly'] == 1][col].std()
        df.loc[:, f'{col}_severity'] = (df[col] - normal_mean) / normal_std

    df = calculate_anomaly_durations(df)

    # Calculating correlation between sensors
    print("Calculating correlations between sensor features...")
    corr_matrix = df[sensor_columns].corr()
    strong_pairs = [(col1, col2) for col1 in sensor_columns for col2 in sensor_columns
                    if corr_matrix.loc[col1, col2] > 0.2 and col1 != col2]

    correlation_scores = []
    for col1, col2 in strong_pairs:
        agreement = ((df[f'{col1}_diff'] > 0) & (df[f'{col2}_diff'] > 0)) | \
                    ((df[f'{col1}_diff'] < 0) & (df[f'{col2}_diff'] < 0))
        correlation_scores.append(agreement.astype(int))

    df.loc[:, 'correlation_behavior'] = pd.DataFrame(correlation_scores).sum(axis=0) / len(strong_pairs)

    # Normalizing features
    print("Normalizing features...")
    weights = {
        'trend_features': 0.3,
        'anomaly_duration': 0.2,
        'severity': 0.3,
        'correlation_behavior': 0.2,
    }

    features_to_normalize = {
        'trend_features': [f'{col}_diff' for col in sensor_columns],
        'anomaly_duration': ['anomaly_duration'],
        'severity': [f'{col}_severity' for col in sensor_columns],
        'correlation_behavior': ['correlation_behavior']
    }

    scaler = StandardScaler()
    for group, columns in features_to_normalize.items():
        df.loc[:, [f'{col}_normalized' for col in columns]] = scaler.fit_transform(df[columns])

    # Calculating weighted score
    print("Calculating weighted scores...")
    df.loc[:, 'weighted_score'] = (
        weights['trend_features'] * df[[f'{col}_normalized' for col in features_to_normalize['trend_features']]].mean(axis=1) +
        weights['anomaly_duration'] * df['anomaly_duration_normalized'] +
        weights['severity'] * df[[f'{col}_normalized' for col in features_to_normalize['severity']]].mean(axis=1) +
        weights['correlation_behavior'] * df['correlation_behavior_normalized']
    )

    # Applying dynamic thresholds
    print("Applying dynamic thresholds for anomaly classification...")
    high_score_threshold = df['weighted_score'].quantile(0.9)
    mid_score_threshold = df['weighted_score'].quantile(0.5)
    high_corr_threshold = df['correlation_behavior'].quantile(0.9)
    mid_corr_threshold = df['correlation_behavior'].quantile(0.5)

    df.loc[:, 'anomaly_class'] = df.apply(
        lambda row: classify_with_dynamic_thresholds(
            row, high_score_threshold, high_corr_threshold,
            mid_score_threshold, mid_corr_threshold
        ), axis=1
    )

    # Applying KMeans clustering
    df = apply_kmeans_clustering(df, features_to_normalize, sensor_columns)

    # Creating directories for saving plots
    os.makedirs("../images", exist_ok=True)

    # PCA Visualization
    # print("Generating PCA visualization...")
    # pca = PCA(n_components=2)
    # proj = pca.fit_transform(df[features_to_normalize['trend_features'] + ['anomaly_duration_normalized', 'correlation_behavior_normalized']])
    # df.loc[:, 'pca1'] = proj[:, 0]
    # df.loc[:, 'pca2'] = proj[:, 1]

    # sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster', palette='Set2')
    # plt.title("K-Means Clustering (PCA Projection)")
    # plt.savefig("../images/pca_kmeans.png")
    # plt.close()

    # sns.scatterplot(data=df, x='pca1', y='pca2', hue='anomaly_class', palette='Set1')
    # plt.title("Heuristic Classification (PCA Projection)")
    # plt.savefig("../images/pca_heuristic.png")
    # plt.close()

    # sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster_mapped_class', palette='Set3')
    # plt.title("KMeans Cluster → Heuristic Label Mapping (PCA Projection)")
    # plt.savefig("../images/pca_cluster_mapping.png")
    # plt.close()

    # # t-SNE Visualization
    # print("Generating t-SNE visualization...")
    # tsne = TSNE(n_components=2, perplexity=30, random_state=0)
    # tsne_proj = tsne.fit_transform(df[features_to_normalize['trend_features'] + ['anomaly_duration_normalized', 'correlation_behavior_normalized']])
    # df.loc[:, 'tsne1'] = tsne_proj[:, 0]
    # df.loc[:, 'tsne2'] = tsne_proj[:, 1]

    # sns.scatterplot(data=df, x='tsne1', y='tsne2', hue='cluster_mapped_class', palette='coolwarm')
    # plt.title("t-SNE: Clusters and Heuristic Mapping")
    # plt.savefig("../images/tsne_cluster_mapping.png")
    # plt.close()

    # Saving the processed data to CSV
    print("Saving processed data to CSV...")
    df.to_csv("../data/processed/classification_dht11_preprocessed.csv", index=False)

    # Print the final anomaly class counts
    print("Final Anomaly Classification Counts:")
    print(df['anomaly_class'].value_counts())

if __name__ == "__main__":
    print("Loading preprocessed data...")
    df_preprocessed = pd.read_csv("../data/processed/sensor_data_preprocessed.csv")
    run_classification(df_preprocessed)
