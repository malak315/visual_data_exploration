## Dataset

### Which dataset are you using? What is it about?

We are using the **Spotify Tracks Dataset (1921-2020)**, which contains audio features and metadata for 586,672 tracks. These tracks span about 100 years of music, and are processed by Spotifys audio analysis algorithms to arrive at ~20 features per track. It also includes artist information and relationships, such as a list of related artists for each artist.

### Where did you get this dataset from? How was the dataset generated?

**Source**: [Kaggle - Spotify Dataset 19212020 600k Tracks](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks)

**Generation Method**:
- **Collection**: Data was collected using the Spotify Web API to query their music catalog
- **Feature Extraction**: Spotify's audio analysis pipeline processes raw audio files to compute perceptual features using machine learning models trained on human annotations
- **Time Period**: The dataset accumulates data from Spotify's catalog spanning 1921 to 2020
- **Processing**: The dataset creator compiled API responses into structured CSV files

### Dataset Size

**Tracks Dataset** (`tracks.csv`):
- **Rows**: 586,672 tracks
- **Columns**: 20 features
- **File Size**: ~180 MB

**Artists Dataset** (`artists.csv`):
- **Rows**: 1,000,000+ artists
- **Columns**: 6 features (id, followers, genres, name, popularity)

**Artist Relationships** (`dict_artists.json`):
- **Entries**: Mappings of top 20 related artists for each artist
- **Purpose**: Cross-reference data for artist popularity analysis
- _Note_: Does not contain cross references for all artists

**Feature Categories**:
- **Audio Features** (13): acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, valence, tempo, duration_ms, key, mode, time_signature
- **Metadata** (7): id, name, artists, release_date, year, popularity, explicit

### What do you want to analyze?

Our primary analysis goals are:

- Analyze clustering patterns and distributions of different music genres based on audio features
- Analyze Trends over time in audio features to see how music characteristics have evolved from 1921 to 2020
- Explore relationships between audio features and popularity metrics
- Validate if audio features can effectively separate genres, moods or popularity levels

### What are you expecting to see?

We are expecting to see related clustering patterns for certain audio features and popularity metrics, i.e. that popularity tends towards certain audio feature profiles. We also expect difficulty in genre seperation, since spotify groups genres in so many categories, that they are probably often overlapping and less useful for clustering. Finally, we expect to see trends over time in certain audio features, e.g., that loudness has increased over the decades.