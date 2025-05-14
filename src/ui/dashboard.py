import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np
from datetime import datetime, timedelta
import random
import glob

# Define emotion colors for visualization
EMOTION_COLORS = {
    "neutral": "#607D8B",
    "calm": "#1E88E5",
    "happy": "#FFB300",
    "sad": "#5E35B1",
    "angry": "#D32F2F",
    "fearful": "#7CB342",
    "disgust": "#00897B",
    "surprised": "#F06292"
}

class EmotionDashboard:
    """Dashboard for emotion analysis visualization"""
    
    def __init__(self):
        """Initialize the dashboard"""
        self.ensure_data_folders()
    
    def ensure_data_folders(self):
        """Ensure required folders exist"""
        os.makedirs("results", exist_ok=True)
        os.makedirs("results/visualizations", exist_ok=True)
    
    def save_analysis_result(self, audio_path, emotion, confidence_scores):
        """Save analysis result to CSV for dashboard visualizations"""
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        # Prepare the data for saving
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename = os.path.basename(audio_path)
        max_confidence = confidence_scores.get(emotion, 0)
        
        # Prepare the row data
        data = {
            "timestamp": timestamp,
            "filename": filename,
            "dominant_emotion": emotion,
            "confidence": max_confidence,
            "audio_path": audio_path
        }
        
        # Add individual confidence scores for each emotion
        for emotion_name, score in confidence_scores.items():
            data[f"{emotion_name}_confidence"] = score
        
        # Create or append to the CSV file
        csv_path = "results/analysis_history.csv"
        
        if os.path.exists(csv_path):
            # Append to existing file
            df = pd.read_csv(csv_path)
            new_row = pd.DataFrame([data])
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            # Create new file
            df = pd.DataFrame([data])
        
        # Save the dataframe
        df.to_csv(csv_path, index=False)
        
    def display_dashboard(self):
        """Display the dashboard with emotion analysis visualizations"""
        st.markdown("## Emotion Analysis Dashboard")
        
        # Check if there are analysis results to display
        if os.path.exists("results/analysis_history.csv"):
            try:
                history_df = pd.read_csv("results/analysis_history.csv")
                if len(history_df) > 0:
                    # Data available, display dashboard
                    self._display_dashboard_with_data(history_df)
                else:
                    # CSV exists but is empty
                    self.display_no_data_dashboard()
            except Exception as e:
                # Error reading or processing the CSV
                st.error(f"Error loading analysis history: {str(e)}")
                self.display_no_data_dashboard()
        else:
            # CSV file doesn't exist
            self.display_no_data_dashboard()
    
    def display_no_data_dashboard(self):
        """Display dashboard when no data is available"""
        with stylable_container(
            key="no_data_container",
            css_styles="""
                {
                    background-color: white;
                    border-radius: 16px;
                    padding: 32px;
                    text-align: center;
                    margin-top: 24px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 20px 25px -5px rgba(0,0,0,0.05);
                    border: 1px solid rgba(226, 232, 240, 0.8);
                }
            """
        ):
            st.markdown("""
            <div style="max-width: 600px; margin: 0 auto;">
                <h2 style="color: #4F46E5; font-weight: 600; margin-bottom: 16px;">No Analysis Data Yet</h2>
                <p style="color: #6B7280; font-size: 1.1rem; margin-bottom: 24px;">
                    Start analyzing audio files to see visualizations and insights in this dashboard.
                </p>
                <div style="font-size: 4rem; margin: 32px 0;">📊</div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.button("Go to Analyze Audio", use_container_width=True)
            
            st.markdown("""
            <div style="max-width: 600px; margin: 24px auto 0 auto;">
                <h3 style="color: #1F2937; font-weight: 600; font-size: 1.2rem; margin-bottom: 16px;">What You'll See Here</h3>
                <div style="display: flex; flex-wrap: wrap; gap: 12px; justify-content: center; margin-bottom: 24px;">
                    <div style="background-color: #F3F4F6; color: #4B5563; padding: 8px 16px; border-radius: 100px; font-size: 0.9rem;">Emotion Distribution</div>
                    <div style="background-color: #F3F4F6; color: #4B5563; padding: 8px 16px; border-radius: 100px; font-size: 0.9rem;">Confidence Trends</div>
                    <div style="background-color: #F3F4F6; color: #4B5563; padding: 8px 16px; border-radius: 100px; font-size: 0.9rem;">Historical Analysis</div>
                    <div style="background-color: #F3F4F6; color: #4B5563; padding: 8px 16px; border-radius: 100px; font-size: 0.9rem;">Emotion Patterns</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def generate_sample_data(self, num_samples=20):
        """Generate sample data for demonstration (for development only)"""
        emotions = ["happy", "sad", "neutral", "angry", "fearful", "calm", "disgust"]
        filenames = ["recording_1.wav", "sample_1.wav", "voice_note.wav", "speech.wav", "audio_clip.wav"]
        
        data = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        for i in range(num_samples):
            # Generate random timestamp within the date range
            random_days = random.randint(0, 30)
            timestamp = end_date - timedelta(days=random_days)
            
            # Select random emotion and filename
            emotion = random.choice(emotions)
            filename = random.choice(filenames)
            
            # Generate confidence scores
            confidence_scores = {}
            # Make the dominant emotion have higher confidence
            main_confidence = random.uniform(70, 95)
            confidence_scores[emotion] = main_confidence
            
            # Generate other confidence scores
            remaining = 100 - main_confidence
            other_emotions = [e for e in emotions if e != emotion]
            for e in other_emotions:
                if e == other_emotions[-1]:
                    # Last emotion gets the remainder
                    confidence_scores[e] = remaining
                else:
                    # Random portion of the remaining confidence
                    score = random.uniform(0, remaining * 0.8)
                    confidence_scores[e] = score
                    remaining -= score
            
            # Create row data
            row = {
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "filename": filename,
                "dominant_emotion": emotion,
                "confidence": confidence_scores[emotion],
                "audio_path": f"uploads/{filename}"
            }
            
            # Add individual confidence scores
            for emotion_name, score in confidence_scores.items():
                row[f"{emotion_name}_confidence"] = score
            
            data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save to CSV
        os.makedirs("results", exist_ok=True)
        df.to_csv("results/analysis_history.csv", index=False)
        
        return df
    
    def _display_dashboard_with_data(self, df):
        """Display dashboard with actual data visualization"""
        # Create tabs for different views
        dashboard_tabs = st.tabs(["📊 Overview", "📈 Trends", "🔍 Details"])
        
        with dashboard_tabs[0]:
            self._display_overview_tab(df)
        
        with dashboard_tabs[1]:
            self._display_trends_tab(df)
        
        with dashboard_tabs[2]:
            self._display_details_tab(df)
    
    def _display_overview_tab(self, df):
        """Display the overview tab with summary statistics and charts"""
        # Show key metrics in cards
        st.markdown("### Key Metrics")
        
        total_analyses = len(df)
        unique_emotions = df['dominant_emotion'].nunique()
        avg_confidence = df['confidence'].mean() if 'confidence' in df.columns else 0
        recent_analyses = len(df[pd.to_datetime(df['timestamp']) >= (datetime.now() - timedelta(days=7))])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self._metric_card("Total Analyses", total_analyses, "📊")
        
        with col2:
            self._metric_card("Unique Emotions", unique_emotions, "🎭")
        
        with col3:
            self._metric_card("Avg. Confidence", f"{avg_confidence:.1f}%", "📈")
        
        with col4:
            self._metric_card("Recent (7 days)", recent_analyses, "📅")
        
        # Emotion distribution chart
        st.markdown("### Emotion Distribution")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            self._emotion_distribution_chart(df)
        
        with col2:
            self._emotion_insights(df)
    
    def _display_trends_tab(self, df):
        """Display the trends tab with time-based visualizations"""
        st.markdown("### Emotion Trends Over Time")
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create daily counts
            df['date'] = df['timestamp'].dt.date
            daily_counts = df.groupby(['date', 'dominant_emotion']).size().reset_index(name='count')
            
            # Create line chart
            fig = px.line(
                daily_counts,
                x='date',
                y='count',
                color='dominant_emotion',
                title='Emotion Trends by Day',
                color_discrete_map={emotion: EMOTION_COLORS.get(emotion, "#607D8B") 
                                  for emotion in daily_counts['dominant_emotion'].unique()}
            )
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Number of Occurrences",
                legend_title="Emotion",
                font=dict(family="Inter, Arial, sans-serif"),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Confidence trends
            st.markdown("### Confidence Score Trends")
            
            # Group by date for average confidence
            confidence_by_date = df.groupby('date')['confidence'].mean().reset_index()
            
            fig = px.line(
                confidence_by_date,
                x='date',
                y='confidence',
                title='Average Confidence Score by Day',
                markers=True
            )
            
            fig.update_traces(line=dict(color='#4F46E5', width=3))
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Average Confidence (%)",
                font=dict(family="Inter, Arial, sans-serif"),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _display_details_tab(self, df):
        """Display the details tab with data table and advanced analytics"""
        st.markdown("### Analysis Records")
        
        # Add filters
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            # Filter by emotion
            emotions = ['All'] + sorted(df['dominant_emotion'].unique().tolist())
            selected_emotion = st.selectbox("Filter by Emotion", emotions)
        
        with col2:
            # Filter by date range
            date_options = ['All Time', 'Last 7 Days', 'Last 30 Days', 'Custom']
            date_filter = st.selectbox("Date Range", date_options)
        
        with col3:
            # Sort options
            sort_options = ['Newest First', 'Oldest First', 'Highest Confidence', 'Lowest Confidence']
            sort_selection = st.selectbox("Sort by", sort_options)
        
        # Apply filters
        filtered_df = df.copy()
        
        # Emotion filter
        if selected_emotion != 'All':
            filtered_df = filtered_df[filtered_df['dominant_emotion'] == selected_emotion]
        
        # Date filter
        if date_filter != 'All Time':
            filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'])
            
            if date_filter == 'Last 7 Days':
                cutoff_date = datetime.now() - timedelta(days=7)
                filtered_df = filtered_df[filtered_df['timestamp'] >= cutoff_date]
            elif date_filter == 'Last 30 Days':
                cutoff_date = datetime.now() - timedelta(days=30)
                filtered_df = filtered_df[filtered_df['timestamp'] >= cutoff_date]
            elif date_filter == 'Custom':
                date_range = st.date_input("Select date range", 
                                          value=(datetime.now() - timedelta(days=7), datetime.now()))
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    filtered_df = filtered_df[(filtered_df['timestamp'].dt.date >= start_date) & 
                                             (filtered_df['timestamp'].dt.date <= end_date)]
        
        # Apply sorting
        if sort_selection == 'Newest First':
            filtered_df = filtered_df.sort_values('timestamp', ascending=False)
        elif sort_selection == 'Oldest First':
            filtered_df = filtered_df.sort_values('timestamp', ascending=True)
        elif sort_selection == 'Highest Confidence':
            filtered_df = filtered_df.sort_values('confidence', ascending=False)
        elif sort_selection == 'Lowest Confidence':
            filtered_df = filtered_df.sort_values('confidence', ascending=True)
        
        # Display the filtered dataframe
        st.dataframe(filtered_df, use_container_width=True)
        
        # Export option
        st.download_button(
            label="Export to CSV",
            data=filtered_df.to_csv(index=False).encode('utf-8'),
            file_name='emotion_analysis_export.csv',
            mime='text/csv',
        )
    
    def _metric_card(self, label, value, icon):
        """Display a metric in a card format"""
        st.markdown(f"""
        <div style="background-color: white; border-radius: 12px; padding: 20px; text-align: center; height: 100%; box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 20px 25px -5px rgba(0,0,0,0.05); border: 1px solid rgba(226, 232, 240, 0.8);">
            <div style="font-size: 1.8rem; margin-bottom: 12px;">{icon}</div>
            <div style="font-size: 2rem; font-weight: 700; color: #4F46E5; margin-bottom: 8px;">{value}</div>
            <div style="font-size: 0.9rem; color: #6B7280;">{label}</div>
        </div>
        """, unsafe_allow_html=True)
    
    def _emotion_distribution_chart(self, df):
        """Create and display emotion distribution chart"""
        emotion_counts = df['dominant_emotion'].value_counts().reset_index()
        emotion_counts.columns = ['emotion', 'count']
        
        colors = {emotion: EMOTION_COLORS.get(emotion, "#607D8B") 
                 for emotion in emotion_counts['emotion']}
        
        fig = px.pie(
            emotion_counts,
            names='emotion',
            values='count',
            color='emotion',
            color_discrete_map=colors,
            hole=0.4,
            title="Emotion Distribution"
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            textfont_size=14
        )
        
        fig.update_layout(
            font=dict(family="Inter, Arial, sans-serif"),
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _emotion_insights(self, df):
        """Display emotion insights"""
        with stylable_container(
            key="emotion_insights",
            css_styles="""
                {
                    background-color: white;
                    border-radius: 12px;
                    padding: 20px;
                    height: 100%;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 20px 25px -5px rgba(0,0,0,0.05);
                    border: 1px solid rgba(226, 232, 240, 0.8);
                }
            """
        ):
            top_emotion = df['dominant_emotion'].value_counts().index[0]
            top_emotion_count = df['dominant_emotion'].value_counts().iloc[0]
            top_emotion_pct = top_emotion_count / len(df) * 100
            
            st.markdown(f"""
            <h4 style="font-weight: 600; color: #1F2937; margin-bottom: 16px;">Emotion Insights</h4>
            
            <div style="margin-bottom: 16px;">
                <div style="font-weight: 500; color: #4B5563; margin-bottom: 8px;">Most Common Emotion</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: {EMOTION_COLORS.get(top_emotion, '#607D8B')}; margin-bottom: 4px;">
                    {top_emotion.capitalize()}
                </div>
                <div style="font-size: 0.9rem; color: #6B7280;">
                    Detected in {top_emotion_count} recordings ({top_emotion_pct:.1f}%)
                </div>
            </div>
            
            <div style="margin-bottom: 16px;">
                <div style="font-weight: 500; color: #4B5563; margin-bottom: 8px;">Emotion Balance</div>
                <div style="height: 8px; background-color: #E5E7EB; border-radius: 4px; margin: 8px 0;">
            """, unsafe_allow_html=True)
            
            # Calculate emotion balance
            positive_emotions = ['happy', 'calm']
            negative_emotions = ['sad', 'angry', 'fearful', 'disgust']
            neutral_emotions = ['neutral']
            
            positive_count = df[df['dominant_emotion'].isin(positive_emotions)].shape[0]
            negative_count = df[df['dominant_emotion'].isin(negative_emotions)].shape[0]
            neutral_count = df[df['dominant_emotion'].isin(neutral_emotions)].shape[0]
            
            total = positive_count + negative_count + neutral_count
            
            positive_pct = positive_count / total * 100 if total > 0 else 0
            negative_pct = negative_count / total * 100 if total > 0 else 0
            neutral_pct = neutral_count / total * 100 if total > 0 else 0
            
            st.markdown(f"""
                    <div style="display: flex; height: 8px; border-radius: 4px; overflow: hidden;">
                        <div style="width: {positive_pct}%; background-color: #22C55E;"></div>
                        <div style="width: {neutral_pct}%; background-color: #64748B;"></div>
                        <div style="width: {negative_pct}%; background-color: #EF4444;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #6B7280; margin-top: 4px;">
                        <span>Positive: {positive_pct:.1f}%</span>
                        <span>Neutral: {neutral_pct:.1f}%</span>
                        <span>Negative: {negative_pct:.1f}%</span>
                    </div>
                </div>
                
                <div>
                    <div style="font-weight: 500; color: #4B5563; margin-bottom: 8px;">Recent Emotions</div>
                    <div style="display: flex; flex-wrap: wrap; gap: 6px;">
            """, unsafe_allow_html=True)
            
            # Display recent emotions as badges
            recent_emotions = df.sort_values('timestamp', ascending=False)['dominant_emotion'].head(5).tolist()
            
            for emotion in recent_emotions:
                color = EMOTION_COLORS.get(emotion, "#607D8B")
                st.markdown(f"""
                <div style="background-color: {color}22; color: {color}; padding: 4px 10px; border-radius: 100px; font-size: 0.8rem;">
                    {emotion.capitalize()}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
