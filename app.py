import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import os

# --- Database Operations Logic ---
# Initialize database for storing user data
def init_db():
    conn = sqlite3.connect('users_data.db') 
    c = conn.cursor()

    # Create users table if it does not exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            interests TEXT,
            goals TEXT,
            skill_level TEXT
        )
    ''')
    # Create progress table if it does not exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS progress (
            progress_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            course_id TEXT,
            completion_status REAL,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    conn.commit()
    return conn

# Function to add a new user
def add_user(conn, username, interests, goals, skill_level):
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, interests, goals, skill_level) VALUES (?, ?, ?, ?)",
                  (username, interests, goals, skill_level))
        conn.commit()
        st.success(f"User '{username}' registered successfully!")
        return c.lastrowid
    except sqlite3.IntegrityError:
        st.error(f"Username '{username}' already exists. Please choose a different username.")
        return None

# Function to get user by username
def get_user(conn, username):
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    return c.fetchone()

# Function to update user progress
def update_progress(conn, user_id, course_id, completion_status):
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO progress (user_id, course_id, completion_status) VALUES (?, ?, ?)",
              (user_id, course_id, completion_status))
    conn.commit()
    st.success(f"Progress for {course_id} updated to {completion_status*100:.0f}%!")

# Function to get user's progress
def get_user_progress(conn, user_id):
    c = conn.cursor()
    c.execute("SELECT course_id, completion_status FROM progress WHERE user_id = ?", (user_id,))
    return c.fetchall()

# --- Data Loading and Preprocessing ---
@st.cache_data  # Cache data to improve performance
def load_course_data(file_path='final_plae_recommendation_data.csv.csv'):
    try:
        # Using latin1 encoding as it resolved previous UnicodeDecodeError
        df = pd.read_csv(file_path, encoding='latin1')

        # Remove trailing empty rows and drop rows with essential nulls
        # Removed 'Domain' from subset as it's not present in the provided CSV snippet
        df.dropna(how='all', inplace=True)
        df.dropna(subset=['course_id', 'title', 'Category', 'skills_covered', 'prerequisites'], inplace=True)

        # Combine relevant features for content-based filtering
        # Removed 'Domain' from features as it's not present
        df['features'] = df.apply(
            lambda x: f"{x['title']} {x['Category']} {x['skills_covered']} {x['prerequisites']}",
            axis=1
        )
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please ensure the file exists in the correct directory.")
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# --- Recommendation Engine Logic ---
@st.cache_resource # Cache resource (TF-IDF model and matrix) for efficient reuse
def get_tfidf_matrix(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(dataframe['features'])
    return tfidf, tfidf_matrix

# Function to recommend courses based on user input
def recommend_courses(user_input, tfidf_model, tfidf_matrix, dataframe, top_n=5):
    if not user_input or dataframe.empty:
        return pd.DataFrame() # Return empty if no input or dataframe is empty

    user_tfidf = tfidf_model.transform([user_input])
    cosine_sim = cosine_similarity(user_tfidf, tfidf_matrix)

    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Ensure we don't try to get more recommendations than available courses
    top_n = min(top_n, len(sim_scores))
    course_indices = [i[0] for i in sim_scores[0:top_n]]

    # Return all relevant features from the dataframe for display
    # Added 'Research_papers', 'Related_articles', 'text_books', and 'Github_repository'
    return dataframe.iloc[course_indices][['course_id', 'title', 'Category', 'difficulty_level', 'skills_covered', 'prerequisites', 'youtube_links', 'Research_papers', 'Related_articles', 'text_books', 'Github_repository']]

# --- Streamlit Application ---
def main():
    st.set_page_config(page_title="AcademIQ - Personalized Learning", layout="wide")

    # Custom styling for the sidebar and main content
    st.markdown("""
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
        <style>
        html, body, [class*="st-emotion"] {
            font-family: 'Poppins', sans-serif;
        }
        /* Main content area background */
        .stApp {
            background-color: #222831; /* Dark Slate Gray */
            color: #eeeeee; /* Off-white text */
        }
        /* Sidebar background */
        .css-1d391kg { 
            background-color: #393e46; /* Slightly Lighter Dark Gray */
        }
        /* Sidebar text color */
        .css-1lcbmhc { /* This class targets the sidebar content */
            color: #eeeeee;
        }
        .stButton>button { /* Button styling */
            background-color: #4CAF50; /* Green */
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            cursor: pointer;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
            transition: all 0.3s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #45a049;
            box-shadow: 3px 3px 8px rgba(0,0,0,0.3);
            transform: translateY(-2px);
        }
        .stRadio > label > div { /* Radio button styling */
            font-size: 18px;
            font-weight: 600; /* Poppins semi-bold */
            color: #eeeeee; /* Off-white */
        }
        .stTextInput, .stTextArea, .stSelectbox { /* Input field styling */
            border-radius: 8px;
            border: 1px solid #76ABAE; /* Teal border */
            padding: 8px;
            background-color: #393e46; /* Match sidebar background */
            color: #eeeeee;
        }
        h1, h2, h3, h4, h5, h6 { /* Header styling */
            color: #76ABAE; /* Teal accent for headers */
            font-weight: 600; /* Poppins semi-bold */
        }
        .stDataFrame { /* DataFrame styling */
            border-radius: 8px;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
            background-color: #393e46; /* Match sidebar background */
            color: #eeeeee;
        }
        .stMarkdown p, .stMarkdown li { /* General markdown text color */
            color: #eeeeee;
        }
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #393e46; /* Match sidebar background */
            color: #eeeeee;
            text-align: center;
            padding: 10px;
            font-size: 0.8em;
            box-shadow: 0 -2px 5px rgba(0,0,0,0.2);
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar Logo and Name
    st.sidebar.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <span style="font-size: 40px; color: #76ABAE;">📚</span>
            <h2 style='color: #76ABAE; margin: 0;'>AcademIQ</h2>
            <p style='font-size: small; color: #eeeeee; margin: 0;'>Your Learning Companion</p>
        </div>
    """, unsafe_allow_html=True)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Profile", "Recommendations", "Progress"])

    # Initializing database
    conn = init_db()

    # Loading course data and TF-IDF matrix
    df = load_course_data()
    if df is None:
        st.stop()  # Stop execution if dataset fails to load

    tfidf, tfidf_matrix = get_tfidf_matrix(df)

    if page == "Home":
        st.markdown("<h1 style='text-align: center; color: #76ABAE;'> AcademIQ 📚 </h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; color: #eeeeee;'>Your Personalized Learning Recommendation Engine</h3>", unsafe_allow_html=True)
        st.write("""
        AcademIQ is designed to make learning personal and meaningful.
        It understands your unique needs, interests, and pace, offering tailored paths to help you grow.
        Discover the right resources, track your progress, and stay motivated on your learning journey!
        """)

        st.markdown("---")
        st.subheader("Discover the World of Data and Quant Finance!")
        # Adjust video size using columns for a minimal appearance
        col1, col2, col3 = st.columns([1, 2, 1]) # Center the video in a narrower column
        with col2:
            st.video("https://youtu.be/t55ndNdmYbo?si=cMV4_pr6QuVjsTbe") 
            st.caption("Video: Master of Quantitative Finance by University of Technology Sydney")
        st.markdown("---")

    elif page == "Profile":
        st.title("User Profile")
        st.write("Create or manage your learning profile.")

        username = st.text_input("Enter your username:")
        if username:
            user_data = get_user(conn, username)
            if user_data:
                st.info(f"Welcome back, {user_data[1]}!")
                st.session_state['current_user_id'] = user_data[0]
                st.session_state['current_username'] = user_data[1]
                st.session_state['current_interests'] = user_data[2]
                st.session_state['current_goals'] = user_data[3]
                st.session_state['current_skill_level'] = user_data[4]

                st.subheader("Your Current Profile:")
                st.write(f"**Username:** {st.session_state['current_username']}")
                st.write(f"**Interests:** {st.session_state['current_interests']}")
                st.write(f"**Goals:** {st.session_state['current_goals']}")
                st.write(f"**Skill Level:** {st.session_state['current_skill_level']}")

            else:
                st.warning("User not found. Please register or check the username.")
                # Allow new registration
                with st.form("new_user_form"):
                    st.subheader("Register New User")
                    new_username = st.text_input("Choose a Username*", key="new_username_input")
                    new_interests = st.text_area("What are your learning interests? (e.g., Data Science, Web Development, Marketing)*")
                    new_goals = st.text_area("What are your learning goals? (e.g., Get a job, Learn a new skill, Pass an exam)*")
                    new_skill_level = st.selectbox("What is your current skill level?", ["Beginner", "Intermediate", "Advanced"])
                    submit_button = st.form_submit_button("Register")

                    if submit_button:
                        if new_username and new_interests and new_goals:
                            user_id = add_user(conn, new_username, new_interests, new_goals, new_skill_level)
                            if user_id:
                                st.session_state['current_user_id'] = user_id
                                st.session_state['current_username'] = new_username
                                st.session_state['current_interests'] = new_interests
                                st.session_state['current_goals'] = new_goals
                                st.session_state['current_skill_level'] = new_skill_level
                                st.success("Registration successful! You can now proceed to get recommendations.")
                            else:
                                st.error("Failed to register user. Please try again.")
                        else:
                            st.error("Please fill in all the required fields marked with *.")
        else:
            st.info("Please enter a username to load or register your profile.")


    elif page == "Recommendations":
        st.title("Your Personalized Course Recommendations")
        st.write("Filter and get tailored course suggestions.")

        # Get unique categories from the dataset
        categories = ['All'] + sorted(df['Category'].unique().tolist())
        selected_category = st.selectbox("Select Category", categories)

        # Filter DataFrame based on selected category
        final_filtered_df = df.copy()
        if selected_category != 'All':
            final_filtered_df = final_filtered_df[final_filtered_df['Category'] == selected_category]

        user_input_keywords = st.text_area("Further refine with keywords (e.g., machine learning, Python, finance)",
                                           help="Enter specific skills or topics to narrow down recommendations.")

        if st.button("Get Recommendations"):
            if not final_filtered_df.empty:
                # Prepare combined input for recommendation engine
                combined_input_for_recommendation = ""
                if selected_category != 'All':
                    combined_input_for_recommendation += selected_category + " "
                combined_input_for_recommendation += user_input_keywords.strip()

                if not combined_input_for_recommendation.strip():
                    st.warning("Please select a category or enter keywords to get recommendations.")
                    return

                try:
                    # Re-fit TF-IDF on the final filtered data for relevant recommendations
                    filtered_tfidf = TfidfVectorizer(stop_words='english')
                    filtered_tfidf_matrix = filtered_tfidf.fit_transform(final_filtered_df['features'])

                    recommendations = recommend_courses(combined_input_for_recommendation, filtered_tfidf, filtered_tfidf_matrix, final_filtered_df)

                    if not recommendations.empty:
                        st.subheader("Recommended Courses")
                        for index, row in recommendations.iterrows():
                            st.markdown(f"**<span style='color: #76ABAE; font-size: 20px;'>{row['title']}</span>**", unsafe_allow_html=True)
                            st.write(f"**Category:** {row['Category']}")
                            st.write(f"**Difficulty Level:** {row['difficulty_level']}")
                            st.write(f"**Skills Covered:** {row['skills_covered']}")
                            st.write(f"**Prerequisites:** {row['prerequisites']}")
                            st.write(f"**YouTube Link:** [Watch Video]({row['youtube_links']})")
                            
                            # Display Research Papers, Related Articles, Text Books, and Github Repository
                            if pd.notna(row['Research_papers']) and row['Research_papers'].strip():
                                st.write(f"**Research Papers:** {row['Research_papers']}")
                            if pd.notna(row['Related_articles']) and row['Related_articles'].strip():
                                st.write(f"**Related Articles:** {row['Related_articles']}")
                            if pd.notna(row['text_books']) and row['text_books'].strip():
                                st.write(f"**Text Books:** {row['text_books']}")
                            if pd.notna(row['Github_repository']) and row['Github_repository'].strip():
                                st.write(f"**GitHub Repository:** [Link]({row['Github_repository']})")
                            
                            st.markdown("---")
                    else:
                        st.warning("No recommendations found for your selection and keywords. Try different criteria.")
                except ValueError as ve:
                    st.warning(f"Not enough data to generate recommendations for the selected filters. Try broadening your selection. Error: {ve}")
            else:
                st.warning("No courses match your selected category. Please adjust your filters.")


    elif page == "Progress":
        st.title("Track Your Learning Progress")
        st.write("View and update your progress on recommended courses.")

        if 'current_user_id' in st.session_state and st.session_state['current_user_id']:
            st.subheader(f"Progress for {st.session_state['current_username']}")
            user_id = st.session_state['current_user_id']
            user_progress = get_user_progress(conn, user_id)

            if user_progress:
                progress_df = pd.DataFrame(user_progress, columns=['Course ID', 'Completion Status'])
                st.dataframe(progress_df)

                st.subheader("Progress Visualization")
                # Create a bar chart for completion status
                # Ensure 'Course ID' is set as index for the chart to label correctly
                st.bar_chart(progress_df.set_index('Course ID'))
            else:
                st.info("No progress tracked yet. Start taking courses!")

            # Allow updating progress
            st.subheader("Update Course Progress")
            # Ensure only courses from the loaded dataframe are available for selection
            course_id_options = ['Select a Course'] + df['course_id'].unique().tolist()
            course_id_to_update = st.selectbox("Select Course ID to update", course_id_options)

            if course_id_to_update != 'Select a Course':
                completion_percentage = st.slider("Completion Percentage", 0, 100, 0)

                if st.button("Update Progress"):
                    update_progress(conn, user_id, course_id_to_update, completion_percentage / 100.0)
            else:
                st.info("Please select a course to update its progress.")
        else:
            st.warning("Please go to the 'Profile' page and log in or register to track your progress.")

    # Footer
    st.markdown("""
        <div class="footer">
            <p>&copy; 2025 AcademIQ. All rights reserved.</p>
            <p>Powered by Laizer</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
