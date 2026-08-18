# AI-Powered Student Feedback Auditor

An NLP-driven data auditing pipeline that classifies student complaints, predicts sentiment, detects critical safety/urgency triggers, and generates action items automatically. Supports batch CSV processing and dashboard analytics.

## Features
* **Multi-Task NLP Pipeline**: Performs category classification, sentiment analysis, urgency alert detection, and action item generation.
* **Streamlit Analytics Dashboard**: Upload CSVs and visualize real-time KPI metrics, complaints distribution, sentiment donut charts, and export processed sheets.
* **Hybrid Execution Modes**:
  * **Cloud Mode (API)**: Leverages Gemini `gemini-1.5-flash` with JSON output schemas for highly structured and accurate multi-task parsing.
  * **Local Offline Mode (Fallback)**: Executes custom rule-based heuristics and text patterns, allowing offline testing with zero costs.

## Tech Stack
* **Frontend**: Streamlit
* **AI API**: Google Generative AI SDK
* **Libraries**: Pandas, Altair, JSON
* **Language**: Python 3.10+

## Setup & Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/nishanth702/college-feedback-classifier.git
   cd college-feedback-classifier
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Obtain Gemini API Key (Optional)**:
   Get a free key from [Google AI Studio](https://aistudio.google.com/) for Cloud API mode.

4. **Launch the Dashboard**:
   ```bash
   streamlit run app.py
   ```
   Open `http://localhost:8501` in your browser.
