# 🚨 CrisisLens

### AI-Powered Emergency Response & Dynamic Evacuation System

CrisisLens is an AI-powered emergency response and evacuation decision-support system designed to analyze emergency situations, retrieve relevant safety knowledge, assess hazards, and recommend a safer feasible evacuation route.

> **Note:** CrisisLens is a prototype decision-support system and must not replace professional emergency authorities.

## 🎯 Problem

During emergencies, responders may need to process different types of information such as incident reports, images, safety procedures, and building layouts. The safest evacuation route may also change when hazards or exits change.

CrisisLens combines AI-based incident understanding with safety knowledge retrieval and graph-based route optimization to support faster emergency decisions.

## 💡 Solution

The system follows this pipeline:

**Incident Report + Emergency Image**
↓
**Multimodal Analysis**
↓
**LLM Incident Understanding**
↓
**RAG Safety Knowledge Retrieval**
↓
**Risk Assessment**
↓
**Building Graph Update**
↓
**Dynamic Route Optimization**
↓
**Recommended Evacuation Route**

## 🧠 Main Technologies

* **Mistral 7B** — emergency incident understanding and structured information extraction
* **BLIP** — emergency image analysis
* **RAG** — retrieval of relevant emergency and evacuation safety information
* **FAISS** — vector similarity search
* **Sentence Transformers** — text embeddings
* **LangChain** — AI/RAG pipeline orchestration
* **NetworkX** — building graph and route optimization
* **Streamlit** — interactive web interface
* **ngrok** — public deployment for demonstration

## ⭐ Key Feature

### Dynamic Graph-Based Evacuation Optimization

Unlike a system that simply recommends the shortest path, CrisisLens represents the building as a graph and considers:

* Route distance
* Hazard severity
* Risk penalties
* Blocked exits
* Available evacuation routes

When the emergency situation changes, the building graph is updated and the recommended route is recalculated.

For example:

```text
Normal situation:
Room 1 → Corridor A → Exit A

Exit A blocked:
Room 1 → Corridor A → Corridor B → Exit B
```

## 🔄 System Workflow

1. The user provides an emergency incident report.
2. An emergency image can optionally be uploaded.
3. Mistral extracts structured incident information.
4. BLIP provides visual information from the uploaded image.
5. RAG retrieves relevant emergency safety knowledge.
6. The system assesses the hazard and updates the building graph.
7. Blocked routes are excluded.
8. Graph-based pathfinding evaluates available routes.
9. CrisisLens displays the recommended evacuation route and building map.

## 📊 Project Components

```text
CrisisLens
│
├── LLM Incident Understanding
├── Multimodal Image Analysis
├── Retrieval-Augmented Generation
├── Vector Database
├── Risk Assessment
├── Building Graph
├── Dynamic Route Optimization
└── Streamlit Interface
```

## 📁 Project Data

The project uses:

* Synthetic emergency incident reports
* Public fire and smoke image data
* Public emergency and evacuation safety information
* A predefined building graph for route optimization

## 🚀 Demo

The system is demonstrated through an interactive Streamlit application.

Users can provide an incident such as:

```text
At 20:05, a critical fire was detected near Room 1.
There are 5 people in the area.
Exit A is blocked and Exit B is available.
```

CrisisLens analyzes the incident and dynamically recommends an available evacuation route.

## 🛠️ Future Improvements

Possible future extensions include:

* Real-time camera integration
* Automatic incident localization
* More detailed floor-plan integration
* Crowd-density estimation
* Video-based emergency analysis
* More advanced multi-factor risk scoring

## ⚠️ Disclaimer

CrisisLens is an educational prototype developed for demonstration purposes. It is not a certified emergency management or life-safety system and should not be used as a substitute for professional emergency services or official evacuation procedures.

## 📄 License

This project is licensed under the MIT License.
