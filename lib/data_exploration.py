import lib.config as config
import json
import pandas as pd
import lib.logger_utils as logger_utils
import matplotlib.pyplot as plt

# Initialize logger
logger = logger_utils.get_logger("data_exploration")

def explore_top20():
    """Explore the top20 related artists dataset."""
    with open(config.FILE_PATHS["top20"], "r") as f:
        top20 = json.load(f)
        counts = []
        for artist, recommended in top20.items():
            counts.append((artist, len(recommended)))

        df = pd.DataFrame(counts, columns=['artist_id', 'reference_count'])
        
        # Log summary statistics
        logger.metric("Total artists in top20", len(top20))
        logger.metric("Average references per artist", df['reference_count'].mean())

        return df

def show_artist_exploration(df_artists):
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))

    # plot artists followers
    axs[0].hist(df_artists.followers, bins=100, color='skyblue', edgecolor='black')
    axs[0].set_title('Distribution of Artist Followers')
    axs[0].set_xlabel('Number of Followers')
    axs[0].set_ylabel('Number of Artists')
    axs[0].set_yscale('log')  # Log scale for better visibility
    axs[0].grid(axis='y', alpha=0.75)

    # plot artists followers
    axs[1].hist(df_artists.popularity, bins=100, color='skyblue', edgecolor='black')
    axs[1].set_title('Distribution of Artist Popularity')
    axs[1].set_xlabel('Popularity')
    axs[1].set_ylabel('Number of Artists')
    axs[1].set_yscale('log')  # Log scale for better visibility
    axs[1].grid(axis='y', alpha=0.75)

    # plot correlation between followers and popularity
    axs[2].scatter(df_artists.followers, df_artists.popularity, color='skyblue', alpha=0.1, edgecolor='black', s=10)
    axs[2].set_title('Correlation between Followers and Popularity')
    axs[2].set_xlabel('Number of Followers')
    axs[2].set_ylabel('Popularity')
    axs[2].set_xscale('log')  # Log scale for better visibility
    axs[2].grid(True)

    plt.show()