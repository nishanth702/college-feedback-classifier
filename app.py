import gradio as gr
from transformers import pipeline

# Load model
classifier = pipeline("text2text-generation", model="google/flan-t5-small")

# Few-shot prompt template
base_prompt = """Classify the following student feedback into one of these categories:
[Academics, Facilities, Administration, Campus Life, Others]
Feedback: "The hostel food is unhygienic."
Category: Facilities
Feedback: "The teachers don’t complete the syllabus."
Category: Academics
Feedback: "The Wi-Fi in the library never works."
Category: Facilities
Feedback: "There’s no communication from the exam office."
Category: Administration
Feedback: "The auditorium is very impressive!"
Category: Campus Life
Feedback: "{}"
Category:"""

# Prediction function
def classify_feedback(feedback_text):
    prompt = base_prompt.format(feedback_text)
    result = classifier(prompt, max_length=10)[0]['generated_text']
    return result.strip()

# Gradio UI
gr.Interface(
    fn=classify_feedback,
    inputs=gr.Textbox(label="Enter Student Feedback", placeholder="e.g. The library is too noisy."),
    outputs=gr.Textbox(label="Predicted Category"),
    title="🎓 College Feedback Classifier",
    description="Classifies student feedback into categories like Academics, Facilities, etc. using FLAN-T5 and few-shot prompting.",
    theme="soft"
).launch()