# 🚨 CrisisLens: An AI-Powered Emergency Response and Dynamic Evacuation System

CrisisLens is an AI-powered emergency response and evacuation decision-support system designed to analyze emergency situations, retrieve relevant safety knowledge, assess hazards, and recommend a safer feasible evacuation route.

The system combines **Large Language Models, Multimodal AI, RAG, Vector Databases, LangChain, Structured Output, Risk Assessment, and Graph-Based Optimization** into an end-to-end emergency response pipeline.

> ⚠️ **Disclaimer:** CrisisLens is a research proof-of-concept and decision-support system. It is not intended to replace professional emergency services, fire-safety systems, or emergency authorities.

---

## 🎯 Problem Statement

During emergencies, critical information is often fragmented across:

- Incident reports
- Images and visual evidence
- Safety manuals and procedures
- Building layouts
- Information about people and blocked exits

At the same time, hazards can change dynamically, meaning that the shortest evacuation route is not always the safest route.

CrisisLens addresses this challenge by combining multimodal analysis, safety knowledge retrieval, risk assessment, and graph-based route optimization to identify and explain a safer feasible evacuation route.

---

## 💡 Solution

CrisisLens receives:

- 📝 An emergency incident report
- 🖼️ An optional emergency image
- 📍 Starting location
- 🚪 Available exits

The system then:

1. Understands the emergency using an LLM.
2. Extracts structured incident information.
3. Analyzes the uploaded image using a vision-language model.
4. Retrieves relevant emergency procedures using RAG.
5. Assesses the risk based on hazard severity and location.
6. Dynamically updates the building graph.
7. Calculates the lowest-cost feasible evacuation route.
8. Visualizes the recommended route through an interactive Streamlit interface.

---

# 🧠 Key Features & Course Concepts Implemented

CrisisLens integrates multiple Generative AI and LLM concepts into one end-to-end application.

## 🤖 Large Language Model (LLM)

**Mistral 7B Instruct** is used to understand natural-language emergency reports.

The LLM extracts:

- Hazard type
- Location
- Severity
- Number of people
- Blocked exits
- Incident time

Example:

```json
{
  "hazard": "fire",
  "location": "Room 1",
  "severity": "medium",
  "people": "5",
  "blocked_exit": "N/A",
  "time": "20:05"
}
```

This converts:

**Natural Language → Structured Incident Information**

---

## 🧠 Prompt Engineering

Task-specific prompts guide the LLM to:

- Identify emergency-related entities.
- Follow predefined severity values.
- Handle missing information.
- Produce machine-readable JSON.
- Return consistent structured responses.

This makes the LLM output easier to process by the rest of the system.

---

## 👁️ Multimodal / Vision AI

**BLIP (Bootstrapping Language-Image Pre-training)** is used to analyze uploaded emergency images.

The model generates a natural-language description of the image and provides additional visual context.

Pipeline:

```text
Emergency Image
      ↓
     BLIP
      ↓
Visual Description
```

The image and incident report are **complementary sources of information**:

- The report provides textual incident context.
- The image provides visual context.

---

## 📚 Retrieval-Augmented Generation (RAG)

CrisisLens uses RAG to retrieve relevant emergency and evacuation safety information from a dedicated knowledge base.

Instead of relying only on the LLM's internal knowledge:

```text
Emergency Situation
        ↓
      Query
        ↓
   Vector Search
        ↓
Relevant Safety Information
```

The system retrieves relevant safety procedures that can support the emergency assessment.

---

## 🔎 Embeddings & Vector Database

**Sentence Transformers (`all-MiniLM-L6-v2`)** are used to convert safety documents into numerical embeddings.

**FAISS** is used as the vector database for semantic similarity search.

Pipeline:

```text
Safety Documents
      ↓
Sentence Transformers
      ↓
Embeddings
      ↓
FAISS
      ↓
Relevant Documents
```

This allows the system to retrieve semantically relevant safety information rather than relying only on keyword matching.

---

## 🔗 LangChain

LangChain components are used to connect and organize the LLM and RAG workflow.

The project uses LangChain for:

- Document processing
- Retrieval
- Embeddings integration
- Structured output parsing
- Connecting different AI components

---

## 📦 Structured Output

CrisisLens uses LangChain's `StructuredOutputParser` to convert the LLM response into structured JSON.

This creates a reliable interface between the LLM and the deterministic parts of the system.

```text
LLM
 ↓
Structured Output Parser
 ↓
JSON
 ↓
Risk Engine
 ↓
Route Optimization
```

---

# ⚠️ Dynamic Risk Assessment

After extracting the incident information, the system converts the emergency situation into risk information for the building graph.

Risk penalties are assigned according to severity:

| Severity | Risk Penalty |
|---|---:|
| Low | 0 |
| Medium | 5 |
| High | 20 |
| Critical | 100 |

Higher severity creates a higher cost for paths affected by the hazard.

Blocked exits are also removed from feasible evacuation paths.

---

# 🗺️ Dynamic Graph-Based Evacuation Optimization

This is the main additional feature beyond a standard LLM/RAG pipeline.

Instead of asking the LLM to simply recommend an evacuation route, CrisisLens represents the building as a **weighted graph**.

### Graph Components

- **Nodes:** Rooms, corridors, and exits
- **Edges:** Possible movement paths
- **Distance:** Movement cost between nodes
- **Risk:** Hazard-related penalty
- **Blocked status:** Whether a path or exit is unavailable

Example:

```text
Room 1
   |
Corridor A
 /        \
Exit A   Corridor B
            |
          Exit B
```

The route cost is calculated using:

```text
Route Cost = Total Distance + Risk Penalty
```

The system then searches for the **lowest-cost feasible path**.

Therefore:

> **The safest route is not necessarily the shortest route.**

A longer route may be selected if the shorter route passes through a higher-risk area.

---

## 🔄 Dynamic Route Recalculation

The evacuation route is not fixed.

If the emergency situation changes, the graph can be updated.

For example:

```text
Exit A → Blocked
```

The system removes the unavailable path and recalculates the route.

Example:

```text
Before:

Room 1 → Corridor A → Exit A


After Exit A becomes blocked:

Room 1 → Corridor A → Corridor B → Exit B
```

This makes the evacuation recommendation dynamic rather than predefined.

---

# 🧩 Decision-Making Architecture

An important design principle in CrisisLens is that **Mistral does not directly choose the evacuation route**.

Instead, responsibilities are separated:

```text
                Incident Report
                       │
                       ▼
                   Mistral 7B
                       │
                       ▼
              Structured Incident
                       │
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
             RAG            Risk Engine
              │                 │
              ▼                 ▼
       Safety Knowledge     Graph Update
                                │
                                ▼
                           NetworkX
                                │
                                ▼
                    Lowest-Cost Feasible Route
                                │
                                ▼
                         Streamlit Dashboard
```

This separation makes the route-selection process more deterministic and explainable.

---

# 🎨 Streamlit GUI

The project provides an interactive Streamlit interface for:

- 📝 Incident reports
- 🖼️ Emergency image uploads
- 📍 Starting location selection
- 🚪 Available exit selection
- 📋 Structured incident information
- 📚 Retrieved safety information
- 🗺️ Evacuation route visualization
- 🤖 Final emergency assessment

---

# 🔄 Complete System Workflow

```text
Incident Report + Image
          │
          ▼
   ┌───────────────┐
   │ Mistral + BLIP│
   └───────┬───────┘
           │
           ▼
 Situation Understanding
           │
           ▼
     Structured Data
           │
           ▼
          RAG
           │
           ▼
 Safety Knowledge Retrieval
           │
           ▼
    Risk Assessment
           │
           ▼
    Building Graph Update
           │
           ▼
 Network-Based Optimization
           │
           ▼
 Safest Feasible Evacuation Route
           │
           ▼
    Streamlit Dashboard
```

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Mistral 7B Instruct** | Emergency report understanding and structured information extraction |
| **BLIP** | Image captioning and visual context |
| **LangChain** | LLM/RAG orchestration and structured output |
| **Sentence Transformers** | Text embeddings |
| **FAISS** | Vector similarity search |
| **NetworkX** | Building graph and route optimization |
| **PyTorch** | Deep learning framework |
| **Transformers** | LLM and vision model integration |
| **Streamlit** | Interactive web GUI |
| **Matplotlib** | Building/route visualization |
| **Hugging Face** | Models and datasets |
| **Kaggle** | GPU-based development and testing |
| **ngrok** | Temporary public access during development |

---

# 📊 Project Data

The project uses multiple types of data:

### 🔥 Fire & Smoke Images

Public fire and smoke image data is used to test the vision component.

### 📝 Incident Reports

Synthetic emergency reports are used to simulate real-world incident information.

The reports contain information such as:

- Emergency type
- Location
- Severity
- Number of people
- Blocked exits
- Time

### 📚 Safety Knowledge

Emergency and evacuation safety information is used as the knowledge base for the RAG pipeline.

### 🏢 Building Graph

A representative building layout is modeled as a graph containing rooms, corridors, and exits.

---

# 🚀 Deployment

During development, CrisisLens was tested in a GPU-enabled Kaggle environment.

The Streamlit application can be exposed temporarily using ngrok.

The project is also structured for containerized deployment using Docker and Hugging Face Spaces.

---

# 🧪 Example Scenario

### Input

```text
At 20:05, a fire was observed near Room 1.
There were 5 people in the area.
Exit A and Exit B are available.
```

### AI Understanding

```text
Hazard: Fire
Location: Room 1
Severity: Medium
People: 5
Blocked Exit: N/A
Time: 20:05
```

### RAG

The system retrieves relevant emergency and evacuation safety guidance.

### Route Optimization

The building graph is updated according to the emergency severity.

NetworkX evaluates the available evacuation paths using:

```text
Distance + Risk
```

### Output

The system displays:

- Emergency assessment
- Retrieved safety information
- Recommended exit
- Recommended evacuation path
- Building map with the selected route

---

# 💡 What Makes CrisisLens Different?

CrisisLens is not simply an LLM chatbot.

It combines:

```text
LLM
+
Vision AI
+
RAG
+
Vector Search
+
Structured Output
+
Risk Assessment
+
Graph Algorithms
+
Interactive Visualization
```

The key innovation is the **dynamic graph-based evacuation optimization layer**, which adds a deterministic decision-making component on top of the Generative AI pipeline.

Instead of asking an LLM:

> "Which exit should I use?"

the system converts the emergency into structured data, applies risk to the building graph, removes blocked paths, and calculates the lowest-cost feasible evacuation route.

This makes the recommendation more **explainable, dynamic, and reproducible**.

---

# 🔮 Future Improvements

Possible future improvements include:

- Real-time video analysis
- Real-time sensor integration
- Automatic building-map generation
- More advanced hazard propagation models
- Multi-floor building support
- Real-time crowd density estimation
- Integration with emergency response systems
- More advanced route optimization algorithms

---

# ⚠️ Disclaimer

CrisisLens is a **research proof-of-concept** developed for educational purposes.

The evacuation recommendations generated by the system should not be treated as authoritative emergency instructions. Real-world emergency decisions must follow professional safety procedures and instructions from qualified emergency authorities.

---

# 📄 License

This project is licensed under the **MIT License**.
