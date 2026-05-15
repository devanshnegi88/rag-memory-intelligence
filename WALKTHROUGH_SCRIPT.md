# Loom Walkthrough Script: Round 2 Intelligence

## Scene 1: Introduction (30 seconds)
- **Visual**: Show the new V2 Dashboard (`streamlit_app_v2.py`).
- **Talking Point**: "Hi everyone, this is the Round 2 extension of our Memory Intelligence platform. Today we've moved beyond simple retrieval into proactive intelligence: tracking persona drift, classifying intent offline, and resolving conflicting memories."

## Scene 2: Adaptive Persona Drift (1 minute)
- **Visual**: Navigate to the **Persona Timeline** tab.
- **Action**: Hover over the line chart.
- **Talking Point**: "Here we track how the user's personality changes over time. You can see the Drift Score fluctuating. Our engine detected a trigger on Feb 5th—'moving to Seattle'—which caused a shift from a 'formal' to a 'curious' tone."

## Scene 3: Offline Intent Classifier (45 seconds)
- **Visual**: Go back to the **Chat** tab and type: "Remind me to call the boss."
- **Talking Point**: "The system now understands *why* you are talking. We built a custom TF-IDF + Logistic Regression model that runs fully offline on CPU. It has <3ms latency and preserves user privacy by staying off the cloud."

## Scene 4: Conflict Resolution (1 minute)
- **Visual**: Ask "Where do I live and what is my job?" (with Seattle/NY data).
- **Talking Point**: "This is our Conflict Resolution engine. Instead of hallucinating, the Resolver flagged the clashes in Occupation and Residence. It then synthesized a merged answer that explicitly warns the user about these contradictory memories."
