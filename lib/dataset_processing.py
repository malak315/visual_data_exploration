import json
import os
from typing import Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from lib.config import FILE_PATHS
from datetime import datetime
import lib.logger_utils as logger_utils


logger = logger_utils.get_logger("dataset_processing")

def add_genre_count(df_artists: pd.DataFrame) -> pd.DataFrame:
    """Count genres per artist. Adds a new column 'genre_count' to df_artists."""
    df_artists = df_artists.copy()
    df_artists['genre_count'] = df_artists['genres'].str.strip('[]').str.split(',').apply(len)
    return df_artists


def add_log_scaled_popularity(df_artists: pd.DataFrame) -> pd.DataFrame:
    """Add log-scaled popularity. Adds a new column 'popularity_log' to df_artists."""
    df_artists = df_artists.copy()
    df_artists['popularity_log'] = np.log1p(df_artists['popularity'])
    return df_artists


def engineer_artist_features(df_artists: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for artists dataframe.
    
    Adds:
    - has_genres: Binary indicator if artist has genres
    - num_genres: Number of genres
    - log_followers: Log-scaled followers
    - followers_per_genre: Followers divided by number of genres
    - popularity_category: Categorical popularity (Niche, Emerging, Popular, Superstar)
    """
    df_featured = df_artists.copy()
    
    # Basic features
    df_featured['has_genres'] = df_featured['genres'].apply(
        lambda x: 1 if x != '[]' and pd.notna(x) else 0
    )
    df_featured['num_genres'] = df_featured['genres'].apply(
        lambda x: len(eval(x)) if x != '[]' and pd.notna(x) else 0
    )
    
    # Follower-based features
    df_featured['log_followers'] = np.log1p(df_featured['followers'])
    df_featured['followers_per_genre'] = df_featured['followers'] / (df_featured['num_genres'] + 1)
    
    # Popularity categories
    df_featured['popularity_category'] = pd.cut(
        df_featured['popularity'],
        bins=[-1, 10, 30, 50, 100],
        labels=['Niche', 'Emerging', 'Popular', 'Superstar']
    )
    
    return df_featured


def add_related_popularity(
    df_artists: pd.DataFrame,
    top20_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Add mean popularity and log mean popularity score of related artists.
    
    Adds:
    - related_popularity_mean: Mean popularity of related artists
    - related_popularity_log_mean: Mean log popularity of related artists
    """
    df_artists = df_artists.copy()
    top20_path = top20_path or FILE_PATHS["top20"]
    
    # Create a lookup dictionary
    artist_id_to_metrics = df_artists.set_index('id')[['popularity', 'popularity_log']].to_dict('index')
    
    related_pop = []
    related_pop_log = []
    
    logger.info(f"Loading related artists data from {top20_path}")
    with open(top20_path, "r") as f:
        top20 = json.load(f)
    
    logger.info(f"Processing {len(top20):,} artist relationships")
    for artist_id, related_ids in tqdm(top20.items(), desc="Processing related artists"):
        pops = [
            artist_id_to_metrics[rid]['popularity']
            for rid in related_ids
            if rid in artist_id_to_metrics
        ]
        pops_log = [
            artist_id_to_metrics[rid]['popularity_log']
            for rid in related_ids
            if rid in artist_id_to_metrics
        ]
        related_pop.append(np.mean(pops) if pops else 0)
        related_pop_log.append(np.mean(pops_log) if pops_log else 0)
    
    df_artists['related_popularity_mean'] = df_artists['id'].map(
        dict(zip(top20.keys(), related_pop))
    )
    df_artists['related_popularity_log_mean'] = df_artists['id'].map(
        dict(zip(top20.keys(), related_pop_log))
    )
    
    # Fill missing values with 0
    df_artists['related_popularity_mean'] = df_artists['related_popularity_mean'].fillna(0)
    df_artists['related_popularity_log_mean'] = df_artists['related_popularity_log_mean'].fillna(0)
    
    return df_artists


# ============================================================================
# Track Feature Engineering Functions
# ============================================================================

def add_name_length(df_tracks: pd.DataFrame) -> pd.DataFrame:
    """Add track name length. Adds a new column 'name_length' to df_tracks."""
    df_tracks = df_tracks.copy()
    df_tracks['name_length'] = df_tracks['name'].str.len()
    return df_tracks


def add_artists_count(df_tracks: pd.DataFrame) -> pd.DataFrame:
    """
    Add number of artists per track.
    
    Adds:
    - artists_list: List of artist names
    - id_artists_list: List of artist IDs
    - num_artists: Number of artists
    """
    df_tracks = df_tracks.copy()
    df_tracks['artists_list'] = df_tracks['artists'].str.strip("[]'\"").str.split("', '")
    df_tracks['id_artists_list'] = df_tracks['id_artists'].str.strip("[]'\"").str.split("', '")
    df_tracks['num_artists'] = df_tracks['artists_list'].apply(len)
    return df_tracks


def add_artist_aggregates(df_tracks: pd.DataFrame, df_artists: pd.DataFrame) -> pd.DataFrame:
    """
    Add aggregated artist metrics to df_tracks.
    
    Adds:
    - artist_popularity_sum/mean
    - artist_popularity_log_sum/mean
    - artist_followers_sum/mean
    - artist_related_pop_mean
    - artist_related_pop_log_mean
    """
    df_tracks = df_tracks.copy()
    
    # Create a mapping dictionary for faster lookup
    artists_metrics = df_artists.set_index('id')[[
        'popularity', 'popularity_log', 'followers', 'genre_count',
        'related_popularity_mean', 'related_popularity_log_mean'
    ]].to_dict('index')
    
    def get_artist_metrics(artist_ids, artist_metrics):
        """Given a list of artist IDs, return aggregated metrics."""
        metrics = {
            'popularity': [],
            'popularity_log': [],
            'followers': [],
            'genre_count': [],
            'related_pop': [],
            'related_pop_log': []
        }
        
        for artist_id in artist_ids:
            if artist_id in artist_metrics:
                artist_data = artist_metrics[artist_id]
                metrics['popularity'].append(artist_data.get('popularity', 0))
                metrics['popularity_log'].append(artist_data.get('popularity_log', 0))
                metrics['followers'].append(artist_data.get('followers', 0))
                metrics['genre_count'].append(artist_data.get('genre_count', 0))
                metrics['related_pop'].append(artist_data.get('related_popularity_mean', 0))
                metrics['related_pop_log'].append(artist_data.get('related_popularity_log_mean', 0))
        
        return metrics
    
    artist_pop_sum = []
    artist_pop_mean = []
    artist_pop_log_sum = []
    artist_pop_log_mean = []
    artist_followers_sum = []
    artist_followers_mean = []
    artist_related_pop_mean = []
    artist_related_pop_log_mean = []
    
    logger.info("Computing artist aggregates for tracks")
    for id_artists in tqdm(df_tracks['id_artists_list'], desc="Processing tracks"):
        metrics = get_artist_metrics(id_artists, artists_metrics)
        
        artist_pop_sum.append(np.nansum(metrics['popularity']) if metrics['popularity'] else 0)
        artist_pop_mean.append(np.nanmean(metrics['popularity']) if metrics['popularity'] else 0)
        artist_pop_log_sum.append(np.nansum(metrics['popularity_log']) if metrics['popularity_log'] else 0)
        artist_pop_log_mean.append(np.nanmean(metrics['popularity_log']) if metrics['popularity_log'] else 0)
        artist_followers_sum.append(np.nansum(metrics['followers']) if metrics['followers'] else 0)
        artist_followers_mean.append(np.nanmean(metrics['followers']) if metrics['followers'] else 0)
        artist_related_pop_mean.append(np.nanmean(metrics['related_pop']) if metrics['related_pop'] else 0)
        artist_related_pop_log_mean.append(np.nanmean(metrics['related_pop_log']) if metrics['related_pop_log'] else 0)
    
    df_tracks['artist_popularity_sum'] = artist_pop_sum
    df_tracks['artist_popularity_mean'] = artist_pop_mean
    df_tracks['artist_popularity_log_sum'] = artist_pop_log_sum
    df_tracks['artist_popularity_log_mean'] = artist_pop_log_mean
    df_tracks['artist_followers_sum'] = artist_followers_sum
    df_tracks['artist_followers_mean'] = artist_followers_mean
    df_tracks['artist_related_pop_mean'] = artist_related_pop_mean
    df_tracks['artist_related_pop_log_mean'] = artist_related_pop_log_mean
    
    return df_tracks


def add_release_date_parsing(df_tracks: pd.DataFrame) -> pd.DataFrame:
    """Parse release_date into datetime objects."""
    df_tracks = df_tracks.copy()
    
    def parse_date(date_str):
        if type(date_str) != str:
            return date_str
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            try:
                return datetime.strptime(date_str, "%Y-%m")
            except ValueError:
                try:
                    return datetime.strptime(date_str, "%Y")
                except ValueError:
                    return None
    
    df_tracks['release_date'] = df_tracks['release_date'].apply(parse_date)
    return df_tracks


def add_release_days(df_tracks: pd.DataFrame) -> pd.DataFrame:
    """Add release date as number of days since 1920-01-01. Adds a new column 'release_days' to df_tracks."""
    df_tracks = df_tracks.copy()
    # Unix timestamp in days (smaller numbers, easier to interpret)
    df_tracks['release_days'] = (
        pd.to_datetime(df_tracks['release_date'], errors='coerce') - pd.Timestamp('1920-01-01')
    ).dt.days
    return df_tracks


# ============================================================================
# Temporal Feature Engineering
# ============================================================================

def add_temporal_encodings(df_tracks: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal encodings: decade, era, year quartile.
    
    Adds:
    - year: Extracted from release_date
    - decade: Decade (e.g., 1980)
    - decade_str: Decade as string (e.g., '1980s')
    - era: Musical era category
    - year_quartile: Year quartile (Q1-Q4)
    """
    df_tracks = df_tracks.copy()
    
    # Extract year from release_date
    if 'release_date' in df_tracks.columns:
        df_tracks['year'] = df_tracks['release_date'].dt.year.astype(int)
    
    if 'year' in df_tracks.columns:
        df_tracks['decade'] = (df_tracks['year'] // 10) * 10
        df_tracks['decade_str'] = df_tracks['decade'].astype(str) + 's'
        
        # Musical eras
        def get_era(year):
            if pd.isna(year):
                return 'Unknown'
            if year < 1950:
                return 'Pre-War & Early (1920-1949)'
            elif year < 1970:
                return 'Golden Age (1950-1969)'
            elif year < 1990:
                return 'Rock/Disco Era (1970-1989)'
            elif year < 2010:
                return 'Modern Era (1990-2009)'
            else:
                return 'Contemporary (2010+)'
        
        df_tracks['era'] = df_tracks['year'].apply(get_era)
        
        # Year quartiles
        df_tracks['year_quartile'] = pd.qcut(
            df_tracks['year'],
            q=4,
            labels=['Q1', 'Q2', 'Q3', 'Q4'],
            duplicates='drop'
        )
    
    return df_tracks


def add_energy_valence_quadrants(df_tracks: pd.DataFrame) -> pd.DataFrame:
    """
    Add mood quadrants based on energy and valence.
    
    Adds:
    - mood_quadrant: Quadrant based on median split (Energetic Happy, Energetic Sad/Angry, Calm Happy, Calm Sad)
    - detailed_mood: Finer-grained categorization using tertiles
    """
    df_tracks = df_tracks.copy()
    
    energy_median = df_tracks['energy'].median()
    valence_median = df_tracks['valence'].median()
    
    def get_mood_quadrant(row):
        if row['energy'] >= energy_median and row['valence'] >= valence_median:
            return 'Energetic Happy'
        elif row['energy'] >= energy_median and row['valence'] < valence_median:
            return 'Energetic Sad/Angry'
        elif row['energy'] < energy_median and row['valence'] >= valence_median:
            return 'Calm Happy'
        else:
            return 'Calm Sad'
    
    df_tracks['mood_quadrant'] = df_tracks.apply(get_mood_quadrant, axis=1)
    
    # Finer-grained emotional categories (tertile split)
    energy_tertiles = df_tracks['energy'].quantile([0.33, 0.67])
    valence_tertiles = df_tracks['valence'].quantile([0.33, 0.67])
    
    def get_detailed_mood(row):
        energy_level = 'High' if row['energy'] > energy_tertiles.iloc[1] else ('Medium' if row['energy'] > energy_tertiles.iloc[0] else 'Low')
        valence_level = 'Positive' if row['valence'] > valence_tertiles.iloc[1] else ('Neutral' if row['valence'] > valence_tertiles.iloc[0] else 'Negative')
        return f"{energy_level} Energy, {valence_level}"
    
    df_tracks['detailed_mood'] = df_tracks.apply(get_detailed_mood, axis=1)
    return df_tracks


def add_musical_characteristic_categories(df_tracks: pd.DataFrame) -> pd.DataFrame:
    """
    Add categorical encodings for musical characteristics.
    
    Adds:
    - danceability_cat: Low/Medium/High Dance
    - acousticness_cat: Electronic/Mixed/Acoustic
    - speechiness_cat: Music/Some Speech/Speech-Heavy
    - instrumentalness_cat: Vocal/Instrumental
    - liveness_cat: Studio/Live
    """
    df_tracks = df_tracks.copy()
    
    # Danceability
    df_tracks['danceability_cat'] = pd.cut(
        df_tracks['danceability'],
        bins=[0, 0.33, 0.67, 1.0],
        labels=['Low Dance', 'Medium Dance', 'High Dance']
    )
    
    # Acousticness
    df_tracks['acousticness_cat'] = pd.cut(
        df_tracks['acousticness'],
        bins=[0, 0.33, 0.67, 1.0],
        labels=['Electronic', 'Mixed', 'Acoustic']
    )
    
    # Speechiness (to identify rap/podcasts)
    df_tracks['speechiness_cat'] = pd.cut(
        df_tracks['speechiness'],
        bins=[0, 0.33, 0.66, 1.0],
        labels=['Music', 'Some Speech', 'Speech-Heavy']
    )
    
    # Instrumentalness
    df_tracks['instrumentalness_cat'] = pd.cut(
        df_tracks['instrumentalness'],
        bins=[0, 0.5, 1.0],
        labels=['Vocal', 'Instrumental']
    )
    
    # Liveness
    df_tracks['liveness_cat'] = pd.cut(
        df_tracks['liveness'],
        bins=[0, 0.8, 1.0],
        labels=['Studio', 'Live']
    )
    
    return df_tracks


def create_combined_categories(df_tracks: pd.DataFrame) -> pd.DataFrame:
    """
    Create combined categorical features from existing categories.
    
    Requires:
    - era (from add_temporal_encodings)
    - mood_quadrant (from add_energy_valence_quadrants)
    - acousticness_cat (from add_musical_characteristic_categories)
    - danceability_cat (from add_musical_characteristic_categories)
    - energy, valence columns
    
    Adds:
    - era_mood: Combination of era and mood_quadrant
    - acoustic_energy: Combination of acousticness_cat and energy level
    - dance_mood: Combination of danceability_cat and valence level
    """
    df_tracks = df_tracks.copy()
    
    # Check for required columns
    required_cols = ['energy', 'valence']
    missing_cols = [col for col in required_cols if col not in df_tracks.columns]
    if missing_cols:
        logger.warning(f"Missing required columns for combined categories: {missing_cols}")
        return df_tracks
    
    energy_median = df_tracks['energy'].median()
    valence_median = df_tracks['valence'].median()
    
    # Combine era with mood (if both exist)
    if 'era' in df_tracks.columns and 'mood_quadrant' in df_tracks.columns:
        df_tracks['era_mood'] = df_tracks['era'].astype(str) + ' - ' + df_tracks['mood_quadrant'].astype(str)
    else:
        logger.warning("Missing 'era' or 'mood_quadrant' columns. Skipping era_mood creation.")
    
    # Combine acousticness with energy (if acousticness_cat exists)
    if 'acousticness_cat' in df_tracks.columns:
        df_tracks['acoustic_energy'] = df_tracks['acousticness_cat'].astype(str) + ' + ' + \
            df_tracks.apply(
                lambda x: 'High Energy' if x['energy'] > energy_median else 'Low Energy',
                axis=1
            )
    else:
        logger.warning("Missing 'acousticness_cat' column. Skipping acoustic_energy creation.")
    
    # Dance + Valence combination (if danceability_cat exists)
    if 'danceability_cat' in df_tracks.columns:
        df_tracks['dance_mood'] = df_tracks['danceability_cat'].astype(str) + ' + ' + \
            df_tracks.apply(
                lambda x: 'Happy' if x['valence'] > valence_median else 'Sad',
                axis=1
            )
    else:
        logger.warning("Missing 'danceability_cat' column. Skipping dance_mood creation.")
    
    return df_tracks
    

def process_datasets(df_tracks, df_artists, cache=True):
    """Process both datasets by applying all feature engineering functions."""
    if cache:
        # Check for cached processed data
        processed_data_dir = "./data/spotify/processed/"
        artists_path = os.path.join(processed_data_dir, "artists_processed.csv")
        tracks_path = os.path.join(processed_data_dir, "tracks_processed.csv")
        if os.path.exists(artists_path) and os.path.exists(tracks_path):
            logger.info("Loading cached processed data")
            logger.info(df_tracks.columns)
            logger.info(df_artists.columns)
            logger.info(df_artists.describe())
            logger.info(df_tracks.describe())
            df_artists = pd.read_csv(artists_path)
            df_tracks = pd.read_csv(tracks_path)
            df_tracks = add_release_date_parsing(df_tracks)
            return df_tracks, df_artists

    logger.info("Processing datasets with feature engineering")
    # Process artists
    df_artists = add_genre_count(df_artists)
    df_artists = add_log_scaled_popularity(df_artists)
    df_artists = add_related_popularity(df_artists)
    
    # Process tracks
    df_tracks = add_name_length(df_tracks)
    df_tracks = add_artists_count(df_tracks)
    df_tracks = add_artist_aggregates(df_tracks, df_artists)
    df_tracks = add_release_date_parsing(df_tracks)
    df_tracks = add_release_days(df_tracks)

    if cache:
        # Save processed data
        processed_data_dir = "./data/spotify/processed/"
        os.makedirs(processed_data_dir, exist_ok=True)
        logger.info(f"Saving processed data to {processed_data_dir}")
        df_artists.to_csv(os.path.join(processed_data_dir, "artists_processed.csv"), index=False)
        df_tracks.to_csv(os.path.join(processed_data_dir, "tracks_processed.csv"), index=False)
    
    logger.info(df_tracks.columns)
    logger.info(df_artists.columns)
    logger.info(df_artists.describe())
    logger.info(df_tracks.describe())
    
    return df_tracks, df_artists