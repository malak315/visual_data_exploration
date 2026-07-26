import pandas as pd
from tqdm import tqdm

def add_temporal_encodings(df_tracks):
    """Add temporal encodings: decade, era, year quartile."""
    # Extract year from release_date
    df_tracks['year'] = df_tracks['release_date'].dt.year.astype(int)
    df_tracks['decade'] = (df_tracks['year'] // 10) * 10
    df_tracks['decade_str'] = df_tracks['decade'].astype(str) + 's'

    # Musical eras
    def get_era(year):
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
        labels=['Q1','Q2', 'Q3', 'Q4']
        )
    
    return df_tracks

def add_energy_valence_quadrants(df_tracks):
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

def add_musical_characteristic_categories(df_tracks):
    """Add categorical encodings for musical characteristics."""
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

def add_popularity_tiers(df_tracks):
    # Percentile-based popularity
    df_tracks['popularity_tier'] = pd.qcut(
        df_tracks['popularity'], 
        q=4, 
        labels=['Unpopular', 'Moderate', 'Popular', 'Very Popular'],
        duplicates='drop'
    )

    # Binary popular/unpopular (median split)
    pop_median = df_tracks['popularity'].median()
    df_tracks['is_popular'] = (df_tracks['popularity'] >= pop_median).map({True: 'Popular', False: 'Unpopular'})

    return df_tracks

def add_genre_binning(df_tracks, df_artists, top_n=15):
    def extract_first_genre(genre_string):
        """Extract the first genre from the genre string"""
        if pd.isna(genre_string) or genre_string == '[]':
            return 'Unknown'
        # Remove brackets and quotes, split by comma
        genres = genre_string.strip("[]").replace("'", "").split(', ')
        return genres[0] if genres and genres[0] else 'Unknown'

    # Create a mapping from artist ID to primary genre
    artist_to_genre = {}
    for _, row in tqdm(df_artists.iterrows(), total=df_artists.shape[0], desc="Mapping artists to genres"):
        artist_to_genre[row['id']] = extract_first_genre(row['genres'])

    # For each track, get the genre from the first artist
    def get_track_genre(id_artists_str):
        """Get genre from first artist in the track"""
        if pd.isna(id_artists_str):
            return 'Unknown'
        artists = id_artists_str.strip("[]'\"").split("', '")
        first_artist = artists[0] if artists else None
        return artist_to_genre.get(first_artist, 'Unknown')

    df_tracks['primary_genre'] = df_tracks['id_artists'].apply(get_track_genre)

    # Get top genres and bin the rest
    genre_counts = df_tracks['primary_genre'].value_counts()
    top_genres = genre_counts.head(top_n).index.tolist()

    df_tracks['genre_binned'] = df_tracks['primary_genre'].apply(
        lambda x: x if x in top_genres else 'Other'
    )
    return df_tracks

def add_genre_binning_by_track_popularity(df_tracks, df_artists, top_n=15):
    """
    Bin genres based on mean track popularity instead of track count.

    Args:
        df_tracks: DataFrame with track information including 'popularity' column
        df_artists: DataFrame with artist information including genres
        top_n: Number of top genres to keep (default: 15)

    Returns:
        df_tracks with 'primary_genre' and 'genre_binned_pop' columns added
    """
    def extract_first_genre(genre_string):
        """Extract the first genre from the genre string"""
        if pd.isna(genre_string) or genre_string == '[]':
            return 'Unknown'
        # Remove brackets and quotes, split by comma
        genres = genre_string.strip("[]").replace("'", "").split(', ')
        return genres[0] if genres and genres[0] else 'Unknown'

    # Create a mapping from artist ID to primary genre
    artist_to_genre = {}
    for _, row in tqdm(df_artists.iterrows(), total=df_artists.shape[0], desc="Mapping artists to genres"):
        artist_to_genre[row['id']] = extract_first_genre(row['genres'])

    # For each track, get the genre from the first artist
    def get_track_genre(id_artists_str):
        """Get genre from first artist in the track"""
        if pd.isna(id_artists_str):
            return 'Unknown'
        artists = id_artists_str.strip("[]'\"").split("', '")
        first_artist = artists[0] if artists else None
        return artist_to_genre.get(first_artist, 'Unknown')

    df_tracks['primary_genre'] = df_tracks['id_artists'].apply(get_track_genre)

    # Calculate mean popularity per genre
    genre_popularity = df_tracks.groupby('primary_genre')['popularity'].mean()

    # Get top N genres by mean popularity
    top_genres = genre_popularity.nlargest(top_n).index.tolist()

    # Bin genres: keep top genres by popularity, rest become 'Other'
    df_tracks['genre_binned_pop'] = df_tracks['primary_genre'].apply(
        lambda x: x if x in top_genres else 'Other'
    )

    return df_tracks

def add_duration_categories(df_tracks):
    """Add duration categories based on duration in minutes."""

    df_tracks['duration_min'] = df_tracks['duration_ms'] / 60000

    # Duration categories
    df_tracks['duration_cat'] = pd.cut(
        df_tracks['duration_min'],
        bins=[0, 2.5, 3.5, 5, float('inf')],
        labels=['Short (<2.5min)', 'Medium (2.5-3.5min)', 'Standard (3.5-5min)', 'Long (>5min)']
    )
    return df_tracks

def add_technical_feature_categories(df_tracks):
    # Loudness categories
    loudness_tertiles = df_tracks['loudness'].quantile([0.33, 0.67])
    df_tracks['loudness_cat'] = pd.cut(
        df_tracks['loudness'],
        bins=[-60, loudness_tertiles.iloc[0], loudness_tertiles.iloc[1], 0],
        labels=['Quiet', 'Medium', 'Loud']
    )

    # Tempo categories
    tempo_tertiles = df_tracks['tempo'].quantile([0.33, 0.67])
    df_tracks['tempo_cat'] = pd.cut(
        df_tracks['tempo'],
        bins=[0, tempo_tertiles.iloc[0], tempo_tertiles.iloc[1], 250],
        labels=['Slow', 'Medium', 'Fast']
    )

    # Key (music theory)
    key_names = {0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F', 
                6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'}
    df_tracks['key_name'] = df_tracks['key'].map(key_names)
    df_tracks['mode_name'] = df_tracks['mode'].map({0: 'Minor', 1: 'Major'})
    df_tracks['key_mode'] = df_tracks['key_name'] + ' ' + df_tracks['mode_name']
    return df_tracks

def create_combined_categories(df_tracks):
    energy_median = df_tracks['energy'].median()
    valence_median = df_tracks['valence'].median()
    # Combine era with mood
    df_tracks['era_mood'] = df_tracks['era'] + ' - ' + df_tracks['mood_quadrant']

    # Combine acousticness with energy
    df_tracks['acoustic_energy'] = df_tracks['acousticness_cat'].astype(str) + ' + ' + \
                                    df_tracks.apply(
                                            lambda x: 'High Energy' if x['energy'] > energy_median else 'Low Energy', 
                                            axis=1
                                        )

    # Dance + Valence combination
    df_tracks['dance_mood'] = df_tracks['danceability_cat'].astype(str) + ' + ' + \
                                    df_tracks.apply(
                                        lambda x: 'Happy' if x['valence'] > valence_median else 'Sad', 
                                        axis=1
                                    )
    return df_tracks

def add_all_categorical_features(df_tracks, df_artists):
    df_tracks = add_temporal_encodings(df_tracks)
    df_tracks = add_energy_valence_quadrants(df_tracks)
    df_tracks = add_musical_characteristic_categories(df_tracks)
    df_tracks = add_popularity_tiers(df_tracks)
    df_tracks = add_genre_binning(df_tracks, df_artists, top_n=15)
    df_tracks = add_duration_categories(df_tracks)
    df_tracks = add_technical_feature_categories(df_tracks)
    df_tracks = create_combined_categories(df_tracks)
    return df_tracks


def get_all_categorical_features():
    """Return list of all categorical encoding column names."""
    return [
        # Temporal
        'decade_str', 'era', 'year_quartile',
        # Mood
        'mood_quadrant', 'detailed_mood',
        # Musical
        'danceability_cat', 'acousticness_cat', 'speechiness_cat',
        'instrumentalness_cat', 'liveness_cat',
        # Popularity
        'popularity_tier', 'is_popular',
        # Genre
        'primary_genre', 'genre_binned',
        # Technical
        'loudness_cat', 'tempo_cat', 'duration_cat', 'key_mode',
        # Combined
        'era_mood', 'acoustic_energy', 'dance_mood'
    ]


def get_categorical_feature_groups():
    """Return dictionary of encoding groups for organized visualization."""
    return {
        'temporal': ['decade_str', 'era', 'year_quartile'],
        'mood': ['mood_quadrant', 'detailed_mood'],
        'musical': ['danceability_cat', 'acousticness_cat', 'speechiness_cat',
                   'instrumentalness_cat', 'liveness_cat'],
        'genre': ['genre_binned'],
        'technical': ['loudness_cat', 'tempo_cat', 'duration_cat'],
        'popularity': ['popularity_tier', 'is_popular'],
        'combined': ['era_mood', 'acoustic_energy', 'dance_mood'],
        'key_insights': ['era', 'mood_quadrant', 'genre_binned',
                        'acousticness_cat', 'danceability_cat', 'popularity_tier']
    }  