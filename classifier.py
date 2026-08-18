import json
import google.generativeai as genai

class FeedbackClassifier:
    def __init__(self, api_key=None):
        self.api_key = api_key
        if api_key:
            genai.configure(api_key=api_key)

    def classify_single(self, text, mode="Cloud API"):
        """Classify a single student feedback string."""
        if not text or not text.strip():
            return {
                "category": "Other",
                "sentiment": "Neutral",
                "urgency": "Normal",
                "action_item": "N/A"
            }

        if mode == "Cloud API" and self.api_key:
            return self._classify_gemini(text)
        else:
            return self._classify_local(text)

    def _classify_gemini(self, text):
        """Query Gemini API using JSON Structured Mode."""
        try:
            model = genai.GenerativeModel("models/gemini-1.5-flash")
            prompt = (
                "You are an expert academic feedback auditor. Analyze the following student feedback "
                "and return a JSON object with these EXACT keys: \n"
                "  - 'category': (must be one of: Academics, Facilities, Placements, Campus Life, Administration, Other)\n"
                "  - 'sentiment': (must be one of: Positive, Negative, Neutral)\n"
                "  - 'urgency': (must be one of: Urgent, Normal - mark as Urgent if there are safety, security, mental health, bullying, theft, food poisoning, or extreme facility issues)\n"
                "  - 'action_item': (a brief 1-sentence action item to resolve the issue)\n\n"
                f"Student Feedback: \"{text}\"\n\n"
                "Return ONLY the raw JSON block without markdown formatting."
            )
            
            # Using generation config to enforce JSON mode
            response = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            data = json.loads(response.text.strip())
            
            # Clean and validate keys
            return {
                "category": data.get("category", "Other"),
                "sentiment": data.get("sentiment", "Neutral"),
                "urgency": data.get("urgency", "Normal"),
                "action_item": data.get("action_item", "Investigate feedback details.")
            }
        except Exception as e:
            # Fallback on API failure
            return self._classify_local(text)

    def _classify_local(self, text):
        """Lightweight rule-based NLP classifier fallback."""
        text_lower = text.lower()
        
        # 1. Sentiment Heuristics
        positive_words = ["good", "excellent", "great", "love", "like", "best", "helpful", "clean", "impressed", "wonderful"]
        negative_words = ["bad", "worst", "unhygienic", "leak", "leakage", "slow", "broken", "dirty", "unusable", "poor", "stolen", "poison", "fail", "stressful", "noise", "noisy"]
        
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)
        
        if pos_count > neg_count:
            sentiment = "Positive"
        elif neg_count > pos_count:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        # 2. Category Heuristics
        categories = {
            "Academics": ["exam", "teacher", "syllabus", "professor", "class", "syllabus", "lecture", "study", "exam", "course"],
            "Facilities": ["hostel", "mess", "food", "library", "wifi", "toilet", "water", "building", "cctv", "leak", "clean", "auditorium"],
            "Placements": ["placement", "job", "interview", "company", "career", "package", "salary"],
            "Campus Life": ["fest", "sports", "ground", "club", "cultural", "event", "canteen", "friend"],
            "Administration": ["fee", "office", "counter", "register", "warden", "admin", "line", "communication", "exam office"]
        }
        
        category = "Other"
        max_hits = 0
        for cat, keywords in categories.items():
            hits = sum(1 for w in keywords if w in text_lower)
            if hits > max_hits:
                max_hits = hits
                category = cat

        # 3. Urgency Heuristics
        urgent_keywords = ["poison", "stolen", "theft", "ragging", "fight", "safety", "emergency", "sick", "wire", "leak", "unhygienic"]
        urgency = "Normal"
        if any(w in text_lower for w in urgent_keywords):
            urgency = "Urgent"

        # 4. Action Item Templates
        if category == "Academics":
            action = "Schedule faculty performance review and coordinate course curriculum gap audits."
        elif category == "Facilities":
            action = "Dispatch maintenance crew to inspect plumbing, food hygiene, or network router integrity."
        elif category == "Placements":
            action = "Instruct Placement Coordinator to publish recruitment schedules and mock interview slots."
        elif category == "Campus Life":
            action = "Review event logistics and request maintenance department to clear flooded sports ground."
        elif category == "Administration":
            action = "Audit office queues and configure online payment options to reduce lobby congestion."
        else:
            action = "Review detailed feedback in student database and allocate support resources."

        return {
            "category": category,
            "sentiment": sentiment,
            "urgency": urgency,
            "action_item": action
        }
