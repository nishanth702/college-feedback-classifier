import streamlit as st
import pandas as pd
import altair as alt
from classifier import FeedbackClassifier

# Page Setup
st.set_page_config(
    page_title="AI College Feedback Auditor",
    page_icon="🎓",
    layout="wide"
)

# Sidebar
st.sidebar.markdown("### ⚙️ Settings")
api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Input Gemini key to run Cloud API mode.")
st.sidebar.markdown("[Get Free Gemini API Key ↗](https://aistudio.google.com/)")

run_mode = "Local Offline"
if api_key:
    run_mode = st.sidebar.selectbox("Execution Mode", ["Cloud API", "Local Offline"])
else:
    st.sidebar.warning("API key missing. Running in Local Offline Mode.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Test Helpers")

# Sample CSV Downloader
sample_csv_data = """feedback
The library Wi-Fi has not been working for the past three days. Please fix it.
Professor Sharma explains machine learning concepts extremely well. Loved the lecture!
Hostel mess serving stale food. Two students had food poisoning yesterday!
Placements office is not communicating the schedule for upcoming interviews.
The college sports ground is flooded with rainwater and is unusable.
Someone stole my laptop from the lab. There are no CCTV cameras in the hallway.
The registration fees counter is very slow. We have to stand in line for 2 hours.
The hostel rooms have a lot of water leakage during rains. It is very unhygienic.
Loved the cultural fest last night! Great management by the student committee.
Semester exams are scheduled with zero gap between papers. Very stressful.
"""
st.sidebar.download_button(
    label="Download Sample CSV",
    data=sample_csv_data,
    file_name="sample_student_feedback.csv",
    mime="text/csv"
)

# Header
st.title("🎓 AI Student Feedback Auditor")
st.markdown("Automate campus operations audits. Categorize feedbacks, predict sentiments, track urgency, and trigger action items.")

# Tabs
tab1, tab2 = st.tabs(["📊 Batch CSV Audit", "🧪 Single Sandbox"])

# Initialize Classifier
clf = FeedbackClassifier(api_key=api_key if api_key else None)

# Tab 1: Batch Upload
with tab1:
    st.subheader("Upload Student Feedbacks CSV")
    uploaded_file = st.file_uploader("Upload feedback file (Must contain a column named 'feedback')", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "feedback" not in df.columns:
            st.error("The CSV file must contain a column named 'feedback'.")
        else:
            st.write("File Preview (First 5 Rows):", df.head())
            
            if st.button("Run AI Feedback Audit", key="run_audit_btn"):
                results = []
                progress_bar = st.progress(0)
                
                # Run classification
                for idx, row in df.iterrows():
                    text = str(row["feedback"])
                    res = clf.classify_single(text, mode=run_mode)
                    results.append(res)
                    progress_bar.progress((idx + 1) / len(df))
                
                # Merge results back
                res_df = pd.DataFrame(results)
                audited_df = pd.concat([df, res_df], axis=1)
                
                # Metrics Cards
                st.markdown("### 📈 Audit Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                total_feedbacks = len(audited_df)
                neg_df = audited_df[audited_df["sentiment"] == "Negative"]
                neg_percent = (len(neg_df) / total_feedbacks) * 100 if total_feedbacks else 0
                urgent_count = len(audited_df[audited_df["urgency"] == "Urgent"])
                
                # Most common category
                cat_counts = audited_df["category"].value_counts()
                top_category = cat_counts.index[0] if not cat_counts.empty else "N/A"

                col1.metric("Total Feedbacks", total_feedbacks)
                col2.metric("Negative Sentiment %", f"{neg_percent:.1f}%")
                col3.metric("Urgent Action Alerts", urgent_count)
                col4.metric("Top Complaint Area", top_category)

                # Charts
                st.markdown("---")
                st.markdown("### 📊 Distribution Analytics")
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    st.write("#### Category Breakdown")
                    cat_chart = alt.Chart(audited_df).mark_bar(color="#3f7a48").encode(
                        x=alt.X("count()", title="Number of Feedbacks"),
                        y=alt.Y("category:N", sort="-x", title="Category")
                    )
                    st.altair_chart(cat_chart, use_container_width=True)

                with chart_col2:
                    st.write("#### Sentiment Split")
                    sentiment_chart = alt.Chart(audited_df).mark_arc(innerRadius=50).encode(
                        theta="count()",
                        color=alt.Color("sentiment:N", scale=alt.Scale(
                            domain=["Positive", "Neutral", "Negative"],
                            range=["#2e7d32", "#ffb300", "#d32f2f"]
                        ))
                    )
                    st.altair_chart(sentiment_chart, use_container_width=True)

                # Data Table
                st.markdown("---")
                st.markdown("### 📄 Audited Feedback Database")
                
                # Filter Controls
                f_col1, f_col2, f_col3 = st.columns(3)
                sel_cat = f_col1.multiselect("Filter by Category", audited_df["category"].unique())
                sel_sent = f_col2.multiselect("Filter by Sentiment", audited_df["sentiment"].unique())
                sel_urg = f_col3.multiselect("Filter by Urgency", audited_df["urgency"].unique())
                
                filtered_df = audited_df.copy()
                if sel_cat:
                    filtered_df = filtered_df[filtered_df["category"].isin(sel_cat)]
                if sel_sent:
                    filtered_df = filtered_df[filtered_df["sentiment"].isin(sel_sent)]
                if sel_urg:
                    filtered_df = filtered_df[filtered_df["urgency"].isin(sel_urg)]

                st.dataframe(filtered_df, use_container_width=True)

                # CSV Download Button
                csv_data = audited_df.to_csv(index=False)
                st.download_button(
                    label="📥 Export Audited Results (CSV)",
                    data=csv_data,
                    file_name="audited_student_feedback.csv",
                    mime="text/csv"
                )

# Tab 2: Single Sandbox
with tab2:
    st.subheader("Try Single Feedback Input")
    text_input = st.text_area(
        "Enter Student Feedback",
        placeholder="e.g. Someone stole my laptop from the library. There are no CCTV cameras here.",
        height=100
    )
    
    if st.button("Analyze Feedback"):
        if not text_input.strip():
            st.warning("Please enter feedback text to classify.")
        else:
            with st.spinner("Classifying..."):
                res = clf.classify_single(text_input, mode=run_mode)
                
                # Display Results in columns
                st.markdown("### 🎯 Classification Results")
                r_col1, r_col2, r_col3 = st.columns(3)
                
                # Style blocks
                category = res["category"]
                sentiment = res["sentiment"]
                urgency = res["urgency"]
                
                r_col1.info(f"**Category:**\n\n### {category}")
                
                if sentiment == "Positive":
                    r_col2.success(f"**Sentiment:**\n\n### Positive")
                elif sentiment == "Negative":
                    r_col2.error(f"**Sentiment:**\n\n### Negative")
                else:
                    r_col2.warning(f"**Sentiment:**\n\n### Neutral")
                    
                if urgency == "Urgent":
                    r_col3.error(f"**Urgency Level:**\n\n### URGENT ALERT 🚨")
                else:
                    r_col3.success(f"**Urgency Level:**\n\n### Normal")
                
                st.markdown("---")
                st.markdown(f"#### 🛠️ AI-Generated Action Item:")
                st.success(f"**{res['action_item']}**")
