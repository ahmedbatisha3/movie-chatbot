import streamlit as st
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches


# =========================================================
# 1. PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Movie Chatbot",
    page_icon="🎬",
    layout="wide"
)


# =========================================================
# 2. LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "/content/movies_cleaned (1)-1.csv"
    )

    # Make sure important columns exist
    for col in ["title", "overview", "genres"]:

        if col not in df.columns:
            df[col] = ""

        df[col] = (
            df[col]
            .fillna("")
            .astype(str)
        )

    return df


df = load_data()


# =========================================================
# 3. LOAD SEMANTIC EMBEDDINGS
# =========================================================

@st.cache_resource
def load_embeddings():

    embeddings = np.load(
        "/content/movie_embeddings.npy"
    )

    return embeddings


embeddings = load_embeddings()


# =========================================================
# 4. CHECK DATA / EMBEDDINGS
# =========================================================

if len(df) != len(embeddings):

    st.error(
        "❌ Dataset and embeddings have different "
        "numbers of movies."
    )

    st.stop()


# =========================================================
# 5. FIND MOVIE INDICES
# =========================================================

def find_movie_indices(movie_title):

    matches = df[
        df["title"].str.casefold()
        == movie_title.strip().casefold()
    ]

    if matches.empty:

        matches = df[
            df["title"].str.contains(
                movie_title.strip(),
                case=False,
                regex=False,
                na=False
            )
        ]

    return matches.index.tolist()


# =========================================================
# 6. SMART MOVIE SEARCH
# =========================================================

def smart_movie_search(movie_title):

    movie_title = movie_title.strip()

    titles = (
        df["title"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    # Exact match
    for title in titles:

        if title.casefold() == movie_title.casefold():
            return title

    # Partial match
    partial_matches = [
        title
        for title in titles
        if movie_title.casefold()
        in title.casefold()
    ]

    if partial_matches:
        return partial_matches[0]

    # Fuzzy matching
    close_matches = get_close_matches(
        movie_title,
        titles,
        n=1,
        cutoff=0.55
    )

    if close_matches:
        return close_matches[0]

    return None


# =========================================================
# 7. EXTRACT MOVIE TITLE
# =========================================================

def extract_movie_title(user_input):

    text = user_input.strip().lower()

    movie_titles = (
        df["title"]
        .dropna()
        .astype(str)
        .tolist()
    )

    # Longest titles first
    movie_titles = sorted(
        movie_titles,
        key=len,
        reverse=True
    )

    # Exact title inside sentence
    for title in movie_titles:

        title_clean = title.strip().lower()

        if len(title_clean) < 3:
            continue

        if title_clean in text:
            return title

    # Try phrases
    phrases = [
        "like ",
        "similar to ",
        "about ",
        "story of ",
        "story about ",
        "recommend ",
        "tell me about "
    ]

    for phrase in phrases:

        if phrase in text:

            possible_title = (
                text.split(
                    phrase,
                    1
                )[1]
                .strip()
            )

            possible_title = (
                possible_title
                .rstrip("?!.,")
            )

            if possible_title:

                result = smart_movie_search(
                    possible_title
                )

                if result:
                    return result

    return None


# =========================================================
# 8. DETECT INTENT
# =========================================================

def detect_intent(user_input):

    text = user_input.lower().strip()

    # Similar movies
    if any(word in text for word in [
        "similar",
        "like",
        "recommend",
        "recommendation",
        "suggest",
        "movies like"
    ]):

        return "similar_movies"


    # Genre
    if any(word in text for word in [
        "genre",
        "action movies",
        "comedy movies",
        "horror movies",
        "romance movies",
        "drama movies",
        "thriller movies",
        "sci-fi movies",
        "science fiction movies",
        "animation movies",
        "adventure movies",
        "crime movies",
        "fantasy movies"
    ]):

        return "genre_search"


    # Top rated
    if any(phrase in text for phrase in [
        "top rated",
        "highest rated",
        "best rated",
        "highest rating",
        "top movies",
        "best movies",
        "highest rated movies"
    ]):

        return "top_rated"


    # Movie information
    if any(word in text for word in [
        "about",
        "story",
        "plot",
        "what is",
        "details",
        "information",
        "info",
        "tell me"
    ]):

        return "movie_info"


    return "unknown"


# =========================================================
# 9. GET MOVIE INFO
# =========================================================

def get_movie_info(movie_title):

    matches = df[
        df["title"].str.casefold()
        == movie_title.strip().casefold()
    ]

    if matches.empty:
        return None

    movie = matches.iloc[0]

    return {
        "title": movie.get(
            "title",
            ""
        ),

        "overview": movie.get(
            "overview",
            ""
        ),

        "release_date": movie.get(
            "release_date",
            ""
        ),

        "rating": movie.get(
            "vote_average",
            ""
        ),

        "vote_count": movie.get(
            "vote_count",
            ""
        ),

        "runtime": movie.get(
            "runtime",
            ""
        ),

        "popularity": movie.get(
            "popularity",
            ""
        ),

        "genres": movie.get(
            "genres",
            ""
        ),

        "poster_url": movie.get(
            "poster_url",
            ""
        ),

        "backdrop_url": movie.get(
            "backdrop_url",
            ""
        ),

        "original_title": movie.get(
            "original_title",
            ""
        ),

        "original_language": movie.get(
            "original_language",
            ""
        )
    }


# =========================================================
# 10. FORMAT MOVIE INFO
# =========================================================

def format_movie_info(movie_info):

    if movie_info is None:

        return (
            "❌ Sorry, I couldn't find this movie "
            "in the dataset."
        )

    poster = movie_info["poster_url"]

    response = ""

    if poster and str(poster) != "nan":

        response += f"""
<div style="
    text-align:center;
    margin-bottom:20px;
">
<img src="{poster}"
     width="250"
     style="
        border-radius:15px;
        box-shadow:0 4px 15px rgba(0,0,0,0.3);
     ">
</div>
"""

    response += f"""
# 🎬 {movie_info['title']}

### 📝 Overview

{movie_info['overview']}

---

🎭 **Genres:** {movie_info['genres']}

📅 **Release Date:** {movie_info['release_date']}

⭐ **Rating:** {movie_info['rating']}

🗳️ **Vote Count:** {movie_info['vote_count']}

⏱️ **Runtime:** {movie_info['runtime']} minutes

🔥 **Popularity:** {movie_info['popularity']}

🌍 **Original Language:** {movie_info['original_language']}

🎞️ **Original Title:** {movie_info['original_title']}
"""

    return response


# =========================================================
# 11. SEMANTIC RECOMMENDATION
# =========================================================

def semantic_recommend(
    movie_title,
    top_n=10
):

    movie_indices = find_movie_indices(
        movie_title
    )

    if not movie_indices:

        return pd.DataFrame()

    movie_index = movie_indices[0]

    # Calculate semantic similarity
    scores = cosine_similarity(
        embeddings[movie_index].reshape(
            1,
            -1
        ),
        embeddings
    ).flatten()

    # Remove selected movie
    scores[movie_index] = -1

    # Sort by similarity
    ranked_indices = np.argsort(
        scores
    )[::-1]

    ranked_indices = ranked_indices[
        :top_n
    ]

    result = df.iloc[
        ranked_indices
    ].copy()

    result["similarity"] = (
        scores[ranked_indices]
    )

    return result.reset_index(
        drop=True
    )


# =========================================================
# 12. FORMAT MOVIE CARD
# =========================================================

def format_movie_card(row):

    title = row.get(
        "title",
        "Unknown"
    )

    rating = row.get(
        "vote_average",
        "N/A"
    )

    year = row.get(
        "release_year",
        "N/A"
    )

    genres = row.get(
        "genres",
        "N/A"
    )

    poster = row.get(
        "poster_url",
        ""
    )

    card = f"""
### 🎬 {title}

⭐ **Rating:** {rating}

📅 **Year:** {year}

🎭 **Genres:** {genres}
"""

    if poster and str(poster) != "nan":

        card += f"""
<img src="{poster}"
     width="180"
     style="
        border-radius:12px;
        margin-top:8px;
        margin-bottom:10px;
     ">
"""

    return card


# =========================================================
# 13. FORMAT RECOMMENDATIONS
# =========================================================

def format_recommendations(
    recommendations,
    movie_title
):

    if recommendations.empty:

        return (
            f"I couldn't find recommendations "
            f"for {movie_title}."
        )

    response = (
        f"## 🎬 Movies similar to "
        f"{movie_title}\n\n"
    )

    for i, row in recommendations.iterrows():

        response += (
            f"### {i + 1}.\n\n"
            f"{format_movie_card(row)}\n\n"
            f"🧠 **Semantic Similarity:** "
            f"`{row['similarity']:.3f}`\n\n"
            f"---\n\n"
        )

    return response


# =========================================================
# 14. SEARCH BY GENRE
# =========================================================

def search_by_genre(
    genre,
    top_n=10
):

    genre = genre.strip().lower()

    matches = df[
        df["genres"]
        .str.lower()
        .str.contains(
            genre,
            regex=False,
            na=False
        )
    ].copy()

    if matches.empty:

        return matches

    matches["vote_average"] = pd.to_numeric(
        matches["vote_average"],
        errors="coerce"
    )

    matches = matches.sort_values(
        by="vote_average",
        ascending=False
    )

    return matches.head(
        top_n
    ).reset_index(
        drop=True
    )


# =========================================================
# 15. EXTRACT GENRE
# =========================================================

def extract_genre(user_input):

    text = user_input.lower()

    genre_map = {

        "action": "Action",

        "adventure": "Adventure",

        "animation": "Animation",

        "comedy": "Comedy",

        "crime": "Crime",

        "documentary": "Documentary",

        "drama": "Drama",

        "family": "Family",

        "fantasy": "Fantasy",

        "history": "History",

        "horror": "Horror",

        "music": "Music",

        "mystery": "Mystery",

        "romance": "Romance",

        "science fiction": "Science Fiction",

        "sci-fi": "Science Fiction",

        "thriller": "Thriller",

        "war": "War",

        "western": "Western"
    }

    for key, value in genre_map.items():

        if key in text:
            return value

    return None


# =========================================================
# 16. FORMAT GENRE RESULTS
# =========================================================

def format_genre_results(
    recommendations,
    genre
):

    if recommendations.empty:

        return (
            f"❌ I couldn't find movies "
            f"for the genre **{genre}**."
        )

    response = (
        f"## 🎭 Top {genre} Movies\n\n"
    )

    for i, row in recommendations.iterrows():

        response += (
            f"### {i + 1}. "
            f"{row.get('title', 'Unknown')}\n\n"
        )

        poster = row.get(
            "poster_url",
            ""
        )

        if poster and str(poster) != "nan":

            response += f"""
<img src="{poster}"
     width="180"
     style="
        border-radius:12px;
        margin-bottom:10px;
     ">
"""

        response += (
            f"\n⭐ **Rating:** "
            f"{row.get('vote_average', 'N/A')}\n\n"

            f"📅 **Year:** "
            f"{row.get('release_year', 'N/A')}\n\n"

            f"🎭 **Genres:** "
            f"{row.get('genres', 'N/A')}\n\n"

            "---\n\n"
        )

    return response


# =========================================================
# 17. TOP RATED MOVIES
# =========================================================

def get_top_rated(
    top_n=10
):

    result = df.copy()

    result["vote_average"] = pd.to_numeric(
        result["vote_average"],
        errors="coerce"
    )

    result["vote_count"] = pd.to_numeric(
        result["vote_count"],
        errors="coerce"
    )

    result = result.dropna(
        subset=["vote_average"]
    )

    # Ignore movies with very few votes
    result = result[
        result["vote_count"] >= 100
    ]

    result = result.sort_values(
        by=[
            "vote_average",
            "vote_count"
        ],
        ascending=[
            False,
            False
        ]
    )

    return result.head(
        top_n
    ).reset_index(
        drop=True
    )


# =========================================================
# 18. FORMAT TOP RATED
# =========================================================

def format_top_rated(
    recommendations
):

    if recommendations.empty:

        return (
            "❌ I couldn't find "
            "top-rated movies."
        )

    response = (
        "## ⭐ Top Rated Movies\n\n"
    )

    for i, row in recommendations.iterrows():

        response += (
            f"### {i + 1}. "
            f"{row.get('title', 'Unknown')}\n\n"
        )

        poster = row.get(
            "poster_url",
            ""
        )

        if poster and str(poster) != "nan":

            response += f"""
<img src="{poster}"
     width="180"
     style="
        border-radius:12px;
        margin-bottom:10px;
     ">
"""

        response += (
            f"\n⭐ **Rating:** "
            f"{row.get('vote_average', 'N/A')}\n\n"

            f"🗳️ **Votes:** "
            f"{row.get('vote_count', 'N/A')}\n\n"

            f"📅 **Year:** "
            f"{row.get('release_year', 'N/A')}\n\n"

            f"🎭 **Genres:** "
            f"{row.get('genres', 'N/A')}\n\n"

            "---\n\n"
        )

    return response


# =========================================================
# 19. TITLE
# =========================================================

st.title("🎬 Movie Chatbot")

st.write(
    "Ask me about movies, find similar movies, "
    "search by genre, or discover top-rated movies! 🍿"
)


# =========================================================
# 20. SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🎬 Movie Features")

    st.write(
        """
### 🤖 What I can do:

🎬 Movie Details

🔎 Smart Movie Search

🎭 Search by Genre

⭐ Top Rated Movies

🧠 Semantic Recommendations

🖼️ Movie Posters
"""
    )

    st.divider()

    st.write(
        f"🎞️ **Movies in database:** "
        f"{len(df):,}"
    )


# =========================================================
# 21. CHAT HISTORY
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"],
            unsafe_allow_html=True
        )


# =========================================================
# 22. CHAT INPUT
# =========================================================

user_input = st.chat_input(
    "Type your question here..."
)


# =========================================================
# 23. PROCESS USER QUESTION
# =========================================================

if user_input:

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):

        st.markdown(user_input)


    with st.chat_message("assistant"):

        intent = detect_intent(
            user_input
        )

        movie_title = extract_movie_title(
            user_input
        )


        # =============================================
        # MOVIE INFO
        # =============================================

        if intent == "movie_info":

            if movie_title is None:

                response = (
                    "❌ I couldn't find a movie "
                    "title in your question.\n\n"
                    "Try:\n\n"
                    "👉 Tell me about Interstellar"
                )

            else:

                movie_info = get_movie_info(
                    movie_title
                )

                response = format_movie_info(
                    movie_info
                )


        # =============================================
        # SEMANTIC SIMILAR MOVIES
        # =============================================

        elif intent == "similar_movies":

            if movie_title is None:

                response = (
                    "❌ I couldn't find a movie "
                    "title in your question.\n\n"
                    "Try:\n\n"
                    "👉 Movies similar to Interstellar"
                )

            else:

                smart_title = smart_movie_search(
                    movie_title
                )

                if smart_title is None:

                    response = (
                        f"❌ I couldn't find "
                        f"**{movie_title}** "
                        f"in the database."
                    )

                else:

                    recommendations = (
                        semantic_recommend(
                            smart_title,
                            top_n=10
                        )
                    )

                    response = (
                        format_recommendations(
                            recommendations,
                            smart_title
                        )
                    )


        # =============================================
        # GENRE SEARCH
        # =============================================

        elif intent == "genre_search":

            genre = extract_genre(
                user_input
            )

            if genre is None:

                response = (
                    "❌ I couldn't identify "
                    "the genre.\n\n"
                    "Try:\n\n"
                    "👉 Give me action movies\n\n"
                    "👉 Show me horror movies\n\n"
                    "👉 Recommend comedy movies"
                )

            else:

                recommendations = (
                    search_by_genre(
                        genre,
                        top_n=10
                    )
                )

                response = (
                    format_genre_results(
                        recommendations,
                        genre
                    )
                )


        # =============================================
        # TOP RATED
        # =============================================

        elif intent == "top_rated":

            recommendations = (
                get_top_rated(
                    top_n=10
                )
            )

            response = (
                format_top_rated(
                    recommendations
                )
            )


        # =============================================
        # UNKNOWN
        # =============================================

        else:

            response = """
## 👋 Hi!

I can help you discover movies 🎬

### 🎬 Movie Details

> Tell me about Interstellar

> What is the story of Inception?

### 🧠 Similar Movies

> Movies similar to Interstellar

> Recommend movies like Inception

### 🔎 Smart Search

> Recommend movies like interstelar

> Tell me about avatr

### 🎭 Genre Search

> Give me action movies

> Show me horror movies

> Recommend comedy movies

### ⭐ Top Rated

> Show me top rated movies
"""


        # =============================================
        # DISPLAY RESPONSE
        # =============================================

        st.markdown(
            response,
            unsafe_allow_html=True
        )


    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })