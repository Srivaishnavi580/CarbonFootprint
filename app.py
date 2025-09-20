import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os
from dotenv import load_dotenv
import joblib
import hashlib
from datetime import datetime
import bcrypt
from pydantic import BaseModel, Field
from google import genai
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage
import base64


def get_base64_of_bin_file(bin_file):
    """
    Encodes a binary file (like an image) into a base64 string.
    """
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    """
    Sets a local PNG image as the background for the Streamlit app.
    """
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed; /* Keeps background fixed during scrolling */
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

 #set_png_as_page_bg('background.png')

tavily_tool = TavilySearch(
    max_results=5,
    topic="general"
)

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Tool creation
tools = [tavily_tool]
# Tool binding
llm_with_tools = gemini_llm.bind_tools(tools)

# graph nodes
def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)  # Executes tool calls

# graph structure
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

# If the LLM asked for a tool, go to ToolNode; else finish
graph.add_conditional_edges("chat_node", tools_condition)

graph.add_edge("tools", "chat_node")

checkpointer = InMemorySaver()
chatbot = graph.compile(checkpointer=checkpointer)


client = genai.Client()

# Load environment variables
load_dotenv("env")

# MongoDB connection
@st.cache_resource
def init_mongodb():
    client = MongoClient(os.environ.get("MONGODB_URI"), server_api=ServerApi('1'))
    db = client["CO2EmissionDB"]
    return db

# Load models and data
@st.cache_resource
def load_models_and_data():
    try:
        # Load pipelines
        transportation_pipeline = joblib.load("transportation/artifacts/prediction_pipeline.joblib")
        diet_pipeline = joblib.load("diet/artifacts/prediction_pipeline.joblib")
        electricity_pipeline = joblib.load("electricity/artifacts/prediction_pipeline.joblib")
        waste_pipeline = joblib.load("waste/artifacts/prediction_pipeline.joblib")

        # Load datasets
        transportation_df = pd.read_csv("transportation/artifacts/data.csv")
        diet_df = pd.read_csv("diet/artifacts/data.csv")
        electricity_df = pd.read_csv("electricity/artifacts/data.csv")
        waste_df = pd.read_csv("waste/artifacts/data.csv")

        return {
            'pipelines': {
                'transportation': transportation_pipeline,
                'diet': diet_pipeline,
                'electricity': electricity_pipeline,
                'waste': waste_pipeline
            },
            'dataframes': {
                'transportation': transportation_df,
                'diet': diet_df,
                'electricity': electricity_df,
                'waste': waste_df
            }
        }
    except Exception as e:
        st.error(f"Error loading models or data: {str(e)}")
        return None

class Co2Emission(BaseModel):
    """Structured output schema for co2 emission recommendation."""
    recommendation: str = Field(description="Recommendations for less co2 emission")
    recycling_tips: str = Field(description="Recycling tips for less co2 emission")
    future_carbon_emissions: float = Field(description="Future carbon emissions value on current predcited co2 emission")
    future_carbon_emission_reason: str = Field(description="Reason for future carbon emissions")

def get_co2_emission_recommendation(d):
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[f"Provide recommendation, tips and future carbon emission value and reason based on provided input and predcited co2 emission: {d}"],
        config={
            "response_mime_type": "application/json",
            "response_schema": Co2Emission,
        },
    )
    return response.parsed.model_dump()

# Generic prediction function
def predict_co2(model_type, sample_data, models_data):
    try:
        pipeline = models_data['pipelines'][model_type]
        data = pd.DataFrame([sample_data])
        prediction = pipeline.predict(data)

        # Convert all predictions to kg CO2 equivalent per unit for standardization
        if model_type == 'transportation':
            # g/km to kg/day (assuming 50km daily travel)
            standardized = round(float(prediction[0]) * 50 / 1000, 2)
        elif model_type == 'diet':
            # kg per food item to kg/day (assuming 1 serving)
            standardized = round(float(prediction[0]), 2)
            standardized = standardized * 10
        elif model_type == 'electricity':
            # tonnes to kg/day (dividing by 365)
            standardized = round(float(prediction[0]) * 1000 / 365, 2)
            standardized = standardized / 1000
        elif model_type == 'waste':
            # yearly emissions to kg/day
            standardized = round(float(prediction[0]) / 365, 2)

        return standardized
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Authentication functions
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def register_user(username, password, db):
    try:
        collection = db["UserCreds"]
        if collection.find_one({"username": username}):
            return False, "Username already exists"

        hashed_password = hash_password(password)
        user_data = {
            "username": username,
            "password": hashed_password,
            "created_at": datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        }
        collection.insert_one(user_data)
        return True, "Registration successful"
    except Exception as e:
        return False, f"Registration failed: {str(e)}"

def authenticate_user(username, password, db):
    try:
        collection = db["UserCreds"]
        user = collection.find_one({"username": username})
        if user and verify_password(password, user["password"]):
            return True
        return False
    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
        return False

def save_prediction(username, model_type, prediction, future_carbon_emissions, user_input, scope_type, db):
    try:
        collection = db["UserHistory"]
        data = {
            "username": username,
            "type": model_type,
            "co2_value": prediction,
            "future_carbon_emissions": future_carbon_emissions,
            "user_input": user_input,
            "scope": scope_type,
            "datetime": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
            "timestamp": datetime.now()
        }
        collection.insert_one(data)
        return True
    except Exception as e:
        st.error(f"Error saving prediction: {str(e)}")
        return False

# UI styling
def apply_custom_css():
    st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }

    .stApp {
        background: linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 100%);
    }

    .css-1d391kg {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .stButton > button {
        background: linear-gradient(45deg, #2e7d32, #4caf50);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(45deg, #1b5e20, #2e7d32);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    .metric-container {
        background: linear-gradient(135deg, #a5d6a7, #c8e6c9);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
    }

    .prediction-result {
        background: linear-gradient(135deg, #66bb6a, #81c784);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.2em;
        font-weight: bold;
        margin: 20px 0;
    }

    h1, h2, h3 {
        color: #1b5e20;
    }
    </style>
    """, unsafe_allow_html=True)

def login_page(db):
    st.title("🌱 CO2 Emission Tracker")
    st.subheader("Login to Your Account")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")

            if submit:
                if authenticate_user(username, password, db):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")

    with tab2:
        with st.form("register_form"):
            new_username = st.text_input("Choose Username")
            new_password = st.text_input("Choose Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            register = st.form_submit_button("Register")

            if register:
                if new_password != confirm_password:
                    st.error("Passwords don't match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    success, message = register_user(new_username, new_password, db)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)

def create_input_form(model_type, df):
    st.subheader(f"🔍 {model_type.title()} CO2 Prediction")

    sample_data = {}

    if model_type == 'transportation':
        col1, col2 = st.columns(2)
        with col1:
            sample_data['Make'] = st.selectbox("Make", df['Make'].unique())
            sample_data['Vehicle Class'] = st.selectbox("Vehicle Class", df['Vehicle Class'].unique())
            sample_data['Fuel Type'] = st.selectbox("Fuel Type", df['Fuel Type'].unique())
        with col2:
            sample_data['Model'] = st.selectbox("Model", df['Model'].unique())
            sample_data['Transmission'] = st.selectbox("Transmission", df['Transmission'].unique())
            sample_data['Fuel Consumption City (L/100 km)'] = st.number_input(
                "Fuel Consumption City (L/100 km)",
                min_value=float(df['Fuel Consumption City (L/100 km)'].min()),
                max_value=float(df['Fuel Consumption City (L/100 km)'].max()),
                value=float(df['Fuel Consumption City (L/100 km)'].mean())
            )

    elif model_type == 'diet':
        col1, col2 = st.columns(2)
        with col1:
            sample_data['Food product'] = st.selectbox("Food Product", df['Food product'].unique())
            sample_data['Feed'] = st.number_input("Feed", min_value=0.0, value=0.0)
            sample_data['Processing'] = st.number_input("Processing", min_value=0.0, value=0.1)
            sample_data['Transport'] = st.number_input("Transport", min_value=0.0, value=0.1)
        with col2:
            sample_data['Packaging'] = st.number_input("Packaging", min_value=0.0, value=0.05)
            sample_data['Retail'] = st.number_input("Retail", min_value=0.0, value=0.02)
            sample_data['Total from Land to Retail'] = st.number_input("Total from Land to Retail", min_value=0.0, value=0.5)

    elif model_type == 'electricity':
        col1, col2 = st.columns(2)
        with col1:
            sample_data['fuel_type'] = st.selectbox("Fuel Type", df['fuel_type'].unique())
            sample_data['region'] = st.selectbox("Region", df['region'].unique())
            sample_data['plant_age_years'] = st.number_input("Plant Age (Years)", min_value=0.0, value=10.0)
            sample_data['efficiency_percent'] = st.number_input("Efficiency (%)", min_value=0.0, max_value=100.0, value=35.0)
            sample_data['capacity_factor_percent'] = st.number_input("Capacity Factor (%)", min_value=0.0, max_value=100.0, value=50.0)
            sample_data['load_factor'] = st.number_input("Load Factor", min_value=0.0, max_value=1.0, value=0.7)
        with col2:
            sample_data['weather_impact_factor'] = st.number_input("Weather Impact Factor", min_value=0.0, max_value=2.0, value=1.0)
            sample_data['maintenance_factor'] = st.number_input("Maintenance Factor", min_value=0.0, max_value=2.0, value=1.0)
            sample_data['grid_demand_mw'] = st.number_input("Grid Demand (MW)", min_value=0.0, value=5000.0)
            sample_data['operating_temp_celsius'] = st.number_input("Operating Temperature (°C)", min_value=-20.0, max_value=50.0, value=20.0)
            sample_data['distance_to_grid_km'] = st.number_input("Distance to Grid (km)", min_value=0.0, value=25.0)
            sample_data['electricity_generated_mwh'] = st.number_input("Electricity Generated (MWh)", min_value=0.0, value=6000.0)

    elif model_type == 'waste':
        col1, col2 = st.columns(2)
        with col1:
            sample_data['Body Type'] = st.selectbox("Body Type", df['Body Type'].unique())
            sample_data['Sex'] = st.selectbox("Sex", df['Sex'].unique())
            sample_data['Diet'] = st.selectbox("Diet", df['Diet'].unique())
            sample_data['How Often Shower'] = st.selectbox("How Often Shower", df['How Often Shower'].unique())
            sample_data['Heating Energy Source'] = st.selectbox("Heating Energy Source", df['Heating Energy Source'].unique())
            sample_data['Transport'] = st.selectbox("Transport", df['Transport'].unique())
            sample_data['Social Activity'] = st.selectbox("Social Activity", df['Social Activity'].unique())
            sample_data['Frequency of Traveling by Air'] = st.selectbox("Frequency of Traveling by Air", df['Frequency of Traveling by Air'].unique())
            sample_data['Waste Bag Size'] = st.selectbox("Waste Bag Size", df['Waste Bag Size'].unique())
            sample_data['Energy efficiency'] = st.selectbox("Energy Efficiency", df['Energy efficiency'].unique())

        with col2:
            sample_data['Monthly Grocery Bill'] = st.number_input("Monthly Grocery Bill", min_value=0, value=200)
            sample_data['Vehicle Monthly Distance Km'] = st.number_input("Vehicle Monthly Distance (km)", min_value=0, value=1000)
            sample_data['Waste Bag Weekly Count'] = st.number_input("Waste Bag Weekly Count", min_value=0, value=3)
            sample_data['How Long TV PC Daily Hour'] = st.number_input("TV/PC Daily Hours", min_value=0, value=5)
            sample_data['How Many New Clothes Monthly'] = st.number_input("New Clothes Monthly", min_value=0, value=3)
            sample_data['How Long Internet Daily Hour'] = st.number_input("Internet Daily Hours", min_value=0, value=4)

            # Binary cooking method inputs
            st.write("Cooking Methods (Check all that apply):")
            sample_data['Recycling_Metal'] = int(st.checkbox("Recycle Metal", value=True))
            sample_data['Recycling_Plastic'] = int(st.checkbox("Recycle Plastic"))
            sample_data['Recycling_Glass'] = int(st.checkbox("Recycle Glass"))
            sample_data['Recycling_Paper'] = int(st.checkbox("Recycle Paper"))
            sample_data['Cooking_Oven'] = int(st.checkbox("Use Oven"))
            sample_data['Cooking_Microwave'] = int(st.checkbox("Use Microwave"))
            sample_data['Cooking_Stove'] = int(st.checkbox("Use Stove", value=True))
            sample_data['Cooking_Airfryer'] = int(st.checkbox("Use Air Fryer"))

    return sample_data

def analytics_page(db):
    st.title("📊 Analytics Dashboard")
    collection = db["UserHistory"]

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        user_filter = st.selectbox("Filter by User",
                                    ['All Users'] + list(collection.distinct("username")))

    with col2:
        type_filter = st.selectbox("Filter by Type",
                                    ['All Types', 'transportation', 'diet', 'electricity', 'waste'])

    with col3:
        date_range = st.date_input("Select Date Range",
                                    value=[datetime.now().replace(day=1), datetime.now()],
                                    key="date_range")

    # Build query
    query = {}
    if user_filter != 'All Users':
        query['username'] = user_filter
    if type_filter != 'All Types':
        query['type'] = type_filter

    # Fetch data
    data = list(collection.find(query))

    if data:
        df = pd.DataFrame(data)
        df['datetime_parsed'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M:%S')

        # Filter by date if range is selected
        if len(date_range) == 2:
            start_date, end_date = date_range
            df = df[(df['datetime_parsed'].dt.date >= start_date) &
                    (df['datetime_parsed'].dt.date <= end_date)]

        if not df.empty:
            # Summary statistics
            st.subheader("📈 Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <h3>{len(df)}</h3>
                    <p>Total Predictions</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <h3>{df['co2_value'].sum():.2f}</h3>
                    <p>Total CO2 (kg/day)</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-container">
                    <h3>{df['co2_value'].mean():.2f}</h3>
                    <p>Average CO2 (kg/day)</p>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                st.markdown(f"""
                <div class="metric-container">
                    <h3>{len(df['username'].unique())}</h3>
                    <p>Active Users</p>
                </div>
                """, unsafe_allow_html=True)

            # Data table
            st.subheader("📋 Recent Predictions")
            display_df = df[['username', 'type', 'co2_value', 'future_carbon_emissions',
                             'scope', 'datetime_parsed']].sort_values('datetime_parsed', ascending=False)
            st.dataframe(display_df, use_container_width=True)

            # User Performance Analysis
            st.subheader("🏆 User Performance Analysis")

            # Calculate user statistics
            user_stats = df.groupby('username').agg({
                'co2_value': ['sum', 'mean', 'count'],
                'datetime_parsed': ['min', 'max']
            }).round(2)

            user_stats.columns = ['Total_CO2', 'Avg_CO2', 'Predictions_Count', 'First_Entry', 'Last_Entry']
            user_stats = user_stats.reset_index()
            user_stats['Days_Active'] = (user_stats['Last_Entry'] - user_stats['First_Entry']).dt.days + 1
            user_stats['Daily_Avg_CO2'] = (user_stats['Total_CO2'] / user_stats['Days_Active']).round(2)

            # Create tabs for different comparison views
            tab1, tab2, tab3, tab4 = st.tabs(["🥇 Top Performers", "📊 User Rankings", "🔥 Activity Levels", "🎯 Efficiency Metrics"])

            with tab1:
                st.subheader("Top 10 Users by Different Metrics")

                col1, col2 = st.columns(2)

                with col1:
                    # Top users by total CO2 (lowest is better for environment)
                    top_eco_friendly = user_stats.nsmallest(10, 'Total_CO2')
                    fig_top_eco = px.bar(top_eco_friendly,
                                        x='username', y='Total_CO2',
                                        title='🌱 Most Eco-Friendly Users (Lowest Total CO2)',
                                        color='Total_CO2',
                                        color_continuous_scale='Greens_r')
                    fig_top_eco.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#1b5e20',
                        xaxis_tickangle=45
                    )
                    st.plotly_chart(fig_top_eco, use_container_width=True)

                with col2:
                    # Most active users by prediction count
                    top_active = user_stats.nlargest(10, 'Predictions_Count')
                    fig_top_active = px.bar(top_active,
                                           x='username', y='Predictions_Count',
                                           title='⚡ Most Active Users (By Predictions)',
                                           color='Predictions_Count',
                                           color_continuous_scale='Blues')
                    fig_top_active.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#1b5e20',
                        xaxis_tickangle=45
                    )
                    st.plotly_chart(fig_top_active, use_container_width=True)

            with tab2:
                st.subheader("Complete User Rankings")

                # User comparison scatter plot
                fig_scatter = px.scatter(user_stats,
                                       x='Predictions_Count', y='Total_CO2',
                                       size='Days_Active',
                                       color='Avg_CO2',
                                       hover_name='username',
                                       title='User Performance Matrix',
                                       labels={'Predictions_Count': 'Number of Predictions',
                                              'Total_CO2': 'Total CO2 Emissions (kg)',
                                              'Avg_CO2': 'Average CO2 per Prediction'})
                fig_scatter.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#1b5e20'
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

                # Ranking table
                ranking_df = user_stats[['username', 'Total_CO2', 'Avg_CO2', 'Predictions_Count', 'Days_Active', 'Daily_Avg_CO2']].copy()
                ranking_df['Eco_Rank'] = ranking_df['Total_CO2'].rank(method='min')
                ranking_df['Activity_Rank'] = ranking_df['Predictions_Count'].rank(method='min', ascending=False)
                ranking_df = ranking_df.sort_values('Total_CO2')

                st.dataframe(ranking_df, use_container_width=True)

            with tab3:
                st.subheader("User Activity Patterns")

                col1, col2 = st.columns(2)

                with col1:
                    # Activity heatmap by user and day
                    if len(df) > 20:
                        df_activity = df.copy()
                        df_activity['date'] = df_activity['datetime_parsed'].dt.date
                        activity_pivot = df_activity.pivot_table(
                            values='co2_value',
                            index='username',
                            columns='date',
                            aggfunc='count',
                            fill_value=0
                        )

                        fig_heatmap = px.imshow(activity_pivot,
                                              title='Daily Activity Heatmap',
                                              color_continuous_scale='Viridis',
                                              aspect='auto')
                        fig_heatmap.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_color='#1b5e20'
                        )
                        st.plotly_chart(fig_heatmap, use_container_width=True)

                with col2:
                    # User activity timeline
                    fig_timeline = px.timeline(user_stats,
                                             x_start='First_Entry',
                                             x_end='Last_Entry',
                                             y='username',
                                             color='Predictions_Count',
                                             title='User Activity Timeline')
                    fig_timeline.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#1b5e20'
                    )
                    st.plotly_chart(fig_timeline, use_container_width=True)

            with tab4:
                st.subheader("User Efficiency & Behavior Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    # Efficiency radar chart (if multiple users)
                    if len(user_stats) >= 3:
                        # Select top 5 users for radar chart
                        top_users = user_stats.nlargest(5, 'Predictions_Count')

                        # Normalize metrics for radar chart (0-100 scale)
                        metrics = ['Total_CO2', 'Avg_CO2', 'Predictions_Count', 'Days_Active']
                        for metric in metrics:
                            max_val = top_users[metric].max()
                            min_val = top_users[metric].min()
                            if max_val != min_val:
                                if metric in ['Total_CO2', 'Avg_CO2']:  # Lower is better
                                    top_users[f'{metric}_norm'] = 100 - ((top_users[metric] - min_val) / (max_val - min_val) * 100)
                                else:  # Higher is better
                                    top_users[f'{metric}_norm'] = (top_users[metric] - min_val) / (max_val - min_val) * 100
                            else:
                                top_users[f'{metric}_norm'] = 50

                        # Create radar chart
                        fig_radar = go.Figure()

                        for _, user in top_users.iterrows():
                            fig_radar.add_trace(go.Scatterpolar(
                                r=[user['Total_CO2_norm'], user['Avg_CO2_norm'],
                                   user['Predictions_Count_norm'], user['Days_Active_norm']],
                                theta=['Eco-Friendly', 'Efficiency', 'Activity', 'Consistency'],
                                fill='toself',
                                name=user['username']
                            ))

                        fig_radar.update_layout(
                            polar=dict(
                                radialaxis=dict(visible=True, range=[0, 100])
                            ),
                            title='User Performance Radar Chart',
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_color='#1b5e20'
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)

                with col2:
                    # User behavior by category
                    user_category_stats = df.groupby(['username', 'type'])['co2_value'].agg(['sum', 'count']).reset_index()
                    user_category_stats.columns = ['username', 'type', 'total_co2', 'count']

                    fig_category = px.treemap(user_category_stats,
                                            path=['username', 'type'],
                                            values='total_co2',
                                            title='CO2 Distribution by User and Category')
                    fig_category.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#1b5e20'
                    )
                    st.plotly_chart(fig_category, use_container_width=True)

            # Visualizations (Original charts)
            st.subheader("📊 Overall Analytics")

            # CO2 by type
            fig1 = px.bar(df.groupby('type')['co2_value'].sum().reset_index(),
                        x='type', y='co2_value',
                        title='CO2 Emissions by Type',
                        color='co2_value',
                        color_continuous_scale='Greens')
            fig1.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#1b5e20'
            )
            st.plotly_chart(fig1, use_container_width=True)

            # Time series
            daily_emissions = df.groupby(df['datetime_parsed'].dt.date)['co2_value'].sum().reset_index()
            fig2 = px.line(daily_emissions, x='datetime_parsed', y='co2_value',
                            title='Daily CO2 Emissions Over Time')
            fig2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#1b5e20'
            )
            st.plotly_chart(fig2, use_container_width=True)

            # User comparison
            if len(df['username'].unique()) > 1:
                user_emissions = df.groupby('username')['co2_value'].sum().reset_index()
                fig3 = px.pie(user_emissions, values='co2_value', names='username',
                            title='CO2 Emissions by User')
                fig3.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#1b5e20'
                )
                st.plotly_chart(fig3, use_container_width=True)

            # Heatmap by type and user
            if len(df) > 10:
                pivot_df = df.pivot_table(values='co2_value', index='username', columns='type', aggfunc='sum', fill_value=0)
                fig4 = px.imshow(pivot_df,
                                title='CO2 Emissions Heatmap (User vs Type)',
                                color_continuous_scale='Greens')
                fig4.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#1b5e20'
                )
                st.plotly_chart(fig4, use_container_width=True)

        else:
            st.info("No data found for the selected filters.")
    else:
        st.info("No prediction data available yet.")

def chatbot_page(username):
    CONFIG = {'configurable': {'thread_id': username}}

    if 'message_history' not in st.session_state:
        st.session_state['message_history'] = []

    if st.sidebar.button("Clear History"):
        st.session_state['message_history'] = []

    # loading the conversation history
    for message in st.session_state['message_history']:
        with st.chat_message(message['role']):
            st.text(message['content'])

    #{'role': 'user', 'content': 'Hi'}
    #{'role': 'assistant', 'content': 'Hi=ello'}

    user_input = st.chat_input('Type here')

    if user_input:

        # first add the message to message_history
        st.session_state['message_history'].append({'role': 'user', 'content': user_input})
        with st.chat_message('user'):
            st.text(user_input)

        response = chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config=CONFIG)

        ai_message = response['messages'][-1].content
        # first add the message to message_history
        st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
        with st.chat_message('assistant'):
            st.text(ai_message)

def dict_to_markdown(data):
    markdown = f"""
## 🌱 Recommendation
> {data.get('recommendation', 'No recommendation provided.')}

---

## ♻️ Recycling Tips
{data.get('recycling_tips', 'No recycling tips provided.')}

---

## 📊 Future Carbon Emissions
**Estimated Emissions:** `{data.get('future_carbon_emissions', 'N/A')} kg CO₂e`

**Reasoning:**
{data.get('future_carbon_emission_reason', 'No reason provided.')}

---
✨ *Making small choices—like composting and buying locally—can have a big impact on reducing your carbon footprint!*
    """
    return markdown.strip()

def home_page():
    # Page title
    st.title("🌍 CO₂ Emission Insights Dashboard")

    # Intro
    st.markdown(
        """
        ## Why Care About CO₂ Emissions?
        Carbon dioxide (CO₂) is one of the major greenhouse gases responsible for **climate change**.
        Every action we take — from the food we eat 🍎, the transport we use 🚗, the electricity we consume ⚡,
        to the waste we generate 🗑️ — contributes to our **carbon footprint**.

        Monitoring and reducing these emissions is **crucial** to slow down global warming, reduce pollution,
        and create a sustainable future for the planet. 🌱
        """
    )

    # Explanation of emission sources
    st.markdown(
        """
        ### 🔍 Key Areas That Contribute to CO₂ Emissions:
        - **🚗 Transportation**: Cars, bikes, and other vehicles emit CO₂ depending on fuel type, efficiency, and usage.
        - **🥗 Diet**: Different foods have different carbon footprints (e.g., beef 🐄 has much higher emissions than apples 🍏).
        - **⚡ Electricity**: Energy generation (coal, gas, renewables) produces varying amounts of CO₂.
        - **🗑️ Waste**: Household habits, recycling, and lifestyle choices also add to emissions.
        """
    )

    # Importance
    st.markdown(
        """
        ### 🌐 Why This Matters
        - CO₂ emissions are directly linked to **global warming** and extreme weather events.
        - Reducing your carbon footprint can improve **air quality** and **public health**.
        - Small lifestyle changes (choosing renewable electricity, reducing food waste, opting for public transport)
          can make a **big difference**.
        """
    )

    # Dashboard hint
    st.markdown(
        """
        ---
        ### 📊 What You Can Do Here
        This app lets you **estimate CO₂ emissions** from different activities using machine learning models:

        1. **Transportation** → Predict CO₂ from your vehicle.
        2. **Diet** → Check the footprint of your food choices.
        3. **Electricity** → Analyze emissions from power generation.
        4. **Waste** → Understand emissions from daily lifestyle habits.

        👉 Use the sidebar to explore each section and calculate your impact!
        """
    )

    st.success("💡 Tip: Awareness is the first step to change — explore the models and see how your choices matter!")

def main():
    st.set_page_config(
        page_title="CO2 Emission Tracker",
        page_icon="🌱",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    apply_custom_css()

    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    # Initialize MongoDB
    db = init_mongodb()

    if not st.session_state.logged_in:
        login_page(db)
        return

    # Load models and data
    models_data = load_models_and_data()
    if models_data is None:
        st.error("Failed to load models and data. Please check file paths.")
        return

    # Sidebar navigation
    st.sidebar.title(f"Welcome, {st.session_state.username}! 🌿")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()

    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Home", "🚗 Transportation", "🥗 Diet", "⚡ Electricity", "🗑️ Waste", "📊 Analytics", "🤖 Chatbot"]
    )
    # Main content
    if page == "🏠 Home":
        home_page()
    elif page == "📊 Analytics":
        analytics_page(db)
    elif page == "🤖 Chatbot":
        chatbot_page(st.session_state.username)
    else:
        model_type = page.split()[1].lower()

        scope_mapping = {}
        scope_mapping['transportation'] = "Scope 1"
        scope_mapping['diet'] = "Scope 3"
        scope_mapping['electricity'] = "Scope 2"
        scope_mapping['waste'] = "Scope 3"

        scope_reasons = {}
        scope_reasons['electricity'] = "Scope 2: Emissions come from the generation of purchased electricity (indirect emissions from energy you use)."
        scope_reasons['transportation'] = "Scope 1: Emissions are direct from vehicles owned or controlled by the entity (mobile fuel combustion under its control)."
        scope_reasons['diet'] = "Scope 3: Emissions are upstream in the value chain (food production, processing, transport, packaging, retail) outside direct control."
        scope_reasons['waste'] = "Scope 3: Emissions from waste generation, treatment, recycling are indirect and result from activities outside direct, owned sources."

        df = models_data['dataframes'][model_type]

        # Create input form
        sample_data = create_input_form(model_type, df)

        col1, col2 = st.columns([3, 1])

        with col1:
            if st.button(f"🔮 Predict CO2 Emissions", type="primary"):
                prediction = predict_co2(model_type, sample_data, models_data)
                sample_data['predcited co2'] = f"{prediction} kg CO2/day"

                if prediction is not None:
                    st.markdown(f"""
                    <div class="prediction-result">
                        <h2>🎯 Prediction Result</h2>
                        <h1>{str(round(prediction,2))} kg CO2/day</h1>
                        <p>Estimated daily CO2 emissions from {model_type}</p>
                        <p>Scope: {scope_mapping[model_type]}</p>
                        <p>Reason: {scope_reasons[model_type]}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander("Sustainability Insights"):
                        d = get_co2_emission_recommendation(sample_data)
                        future_carbon_emissions = d['future_carbon_emissions']
                        st.markdown(dict_to_markdown(d))

                    # Store prediction in session state for saving
                    st.session_state.last_prediction = {
                        'type': model_type,
                        'value': prediction,
                        'future_carbon_emissions': future_carbon_emissions,
                        'input': sample_data,
                        'scope': scope_mapping[model_type]
                    }

        with col2:
            if 'last_prediction' in st.session_state and st.session_state.last_prediction['type'] == model_type:
                if st.button("💾 Save Prediction", type="secondary"):
                    if save_prediction(
                        st.session_state.username,
                        st.session_state.last_prediction['type'],
                        st.session_state.last_prediction['value'],
                        st.session_state.last_prediction['future_carbon_emissions'],
                        st.session_state.last_prediction['input'],
                        st.session_state.last_prediction['scope'],
                        db
                    ):
                        st.success("✅ Prediction saved successfully!")
                    else:
                        st.error("❌ Failed to save prediction")

if __name__ == "__main__":
    main()
