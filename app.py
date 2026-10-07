import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import streamlit as st
import torch
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="CrisisLens", page_icon="🚨", layout="wide")

st.title("🚨 CrisisLens")
st.subheader("AI-Powered Emergency Response and Dynamic Evacuation System")
st.warning(
    "⚠️ CrisisLens is a research PoC and decision-support system. "
    "It is not a replacement for emergency authorities or professional fire/life-safety systems."
)

BASE_DIR = "/kaggle/working/CrisisLens"
VECTORSTORE_PATH = os.path.join(BASE_DIR, "vectorstore")


@st.cache_resource
def load_mistral():
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


@st.cache_resource
def load_blip():
    model_name = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    if torch.cuda.is_available():
        model = model.to("cuda")
    return processor, model


@st.cache_resource
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        VECTORSTORE_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )


with st.spinner("Loading AI models and knowledge base..."):
    tokenizer, model = load_mistral()
    vision_processor, vision_model = load_blip()
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


response_schemas = [
    ResponseSchema(name="hazard", description="Type of hazard such as fire, smoke, gas_leak, flood, electrical_hazard, or blocked_exit."),
    ResponseSchema(name="location", description="Location where the incident occurred."),
    ResponseSchema(name="severity", description="Severity: low, medium, high, or critical."),
    ResponseSchema(name="people", description="Number of people affected or present."),
    ResponseSchema(name="blocked_exit", description="Blocked exit if mentioned, otherwise N/A."),
    ResponseSchema(name="time", description="Incident time if mentioned, otherwise N/A.")
]
parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()


def create_building_graph():
    graph = nx.Graph()
    graph.add_nodes_from([
        "Room 1", "Room 2", "Room 3",
        "Corridor A", "Corridor B",
        "Exit A", "Exit B"
    ])
    graph.add_edge("Room 1", "Corridor A", distance=2)
    graph.add_edge("Room 2", "Corridor A", distance=3)
    graph.add_edge("Room 3", "Corridor B", distance=3)
    graph.add_edge("Corridor A", "Exit A", distance=2)
    graph.add_edge("Corridor A", "Corridor B", distance=2)
    graph.add_edge("Corridor B", "Exit B", distance=2)
    for u, v, data in graph.edges(data=True):
        data["risk"] = "low"
        data["blocked"] = False
    return graph


building_graph = create_building_graph()

RISK_PENALTIES = {"low": 0, "medium": 5, "high": 20, "critical": 100}


def apply_incident_to_graph(graph, location, severity, blocked_exit):
    graph = graph.copy()
    severity = str(severity).lower()
    if severity not in RISK_PENALTIES:
        severity = "medium"
    if location in graph.nodes:
        for neighbor in graph.neighbors(location):
            graph[location][neighbor]["risk"] = severity
    if blocked_exit in graph.nodes:
        for neighbor in list(graph.neighbors(blocked_exit)):
            graph[blocked_exit][neighbor]["blocked"] = True
    return graph


def find_safest_route(graph, start_location, available_exits):
    graph = graph.copy()
    graph.remove_edges_from([
        (u, v) for u, v, data in graph.edges(data=True)
        if data.get("blocked", False)
    ])

    for u, v, data in graph.edges(data=True):
        data["cost"] = data.get("distance", 1) + RISK_PENALTIES.get(
            data.get("risk", "low"), 0
        )

    best_path, best_cost, best_exit = None, float("inf"), None

    for exit_node in available_exits:
        if exit_node not in graph.nodes:
            continue
        try:
            path = nx.shortest_path(
                graph, source=start_location,
                target=exit_node, weight="cost"
            )
            cost = nx.path_weight(graph, path, weight="cost")
            if cost < best_cost:
                best_path, best_cost, best_exit = path, cost, exit_node
        except nx.NetworkXNoPath:
            continue

    return best_path, best_cost, best_exit


def draw_building_map(graph, route=None):
    positions = {
        "Room 1": (0, 2), "Room 2": (0, 0), "Room 3": (4, 0),
        "Corridor A": (2, 2), "Corridor B": (4, 2),
        "Exit A": (2, 4), "Exit B": (6, 2)
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    node_colors = []
    for node in graph.nodes:
        if route and node in route:
            node_colors.append("limegreen")
        elif node.startswith("Exit"):
            node_colors.append("orange")
        elif node.startswith("Room"):
            node_colors.append("lightblue")
        else:
            node_colors.append("lightgray")

    nx.draw_networkx_nodes(graph, positions, node_color=node_colors, node_size=2200, ax=ax)
    nx.draw_networkx_labels(graph, positions, font_size=9, font_weight="bold", ax=ax)

    edge_colors = [
        "red" if data.get("blocked", False) else "black"
        for _, _, data in graph.edges(data=True)
    ]
    nx.draw_networkx_edges(graph, positions, edge_color=edge_colors, width=2, ax=ax)

    if route:
        route_edges = list(zip(route[:-1], route[1:]))
        nx.draw_networkx_edges(
            graph, positions, edgelist=route_edges,
            edge_color="green", width=5, ax=ax
        )

    ax.set_title("Building Evacuation Map")
    ax.axis("off")
    return fig


def analyze_image(image):
    image = image.convert("RGB")
    inputs = vision_processor(images=image, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
    with torch.no_grad():
        output = vision_model.generate(**inputs, max_new_tokens=50)
    return vision_processor.decode(output[0], skip_special_tokens=True)


def extract_incident(text):
    prompt = f"""
You are an emergency incident information extraction system.

Extract:
- hazard
- location
- severity
- people
- blocked_exit
- time

Allowed severity values: low, medium, high, critical.
If unavailable, use "N/A".
Return ONLY valid JSON.

{format_instructions}

Incident report:
{text}
"""
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {key: value.to("cuda") for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.1,
            do_sample=False
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    try:
        return parser.parse(generated_text)
    except Exception:
        try:
            start = generated_text.find("{")
            end = generated_text.rfind("}") + 1
            return json.loads(generated_text[start:end])
        except Exception:
            return {
                "hazard": "unknown",
                "location": "N/A",
                "severity": "medium",
                "people": "N/A",
                "blocked_exit": "N/A",
                "time": "N/A"
            }


def retrieve_safety_information(query):
    return retriever.invoke(query)


st.sidebar.header("🚨 Incident Configuration")

incident_text = st.text_area(
    "Incident Report",
    placeholder=(
        "Example:\n"
        "At 20:05, a fire was observed near Room 1. "
        "There were 5 people in the area. "
        "Exit A and Exit B are available."
    ),
    height=150
)

uploaded_image = st.file_uploader(
    "Upload Emergency Image",
    type=["jpg", "jpeg", "png", "webp"]
)

start_location = st.selectbox(
    "Starting Location",
    ["Room 1", "Room 2", "Room 3"]
)

available_exits = st.multiselect(
    "Available Exits",
    ["Exit A", "Exit B"],
    default=["Exit A", "Exit B"]
)


if st.button("🚨 Analyze Emergency", type="primary"):

    if not incident_text:
        st.error("Please enter an incident report.")
        st.stop()

    if uploaded_image:
        st.subheader("👁️ Visual Analysis")
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Emergency Image", use_container_width=True)

        with st.spinner("Analyzing image..."):
            visual_description = analyze_image(image)

        st.info(f"BLIP Description: {visual_description}")

    with st.spinner("Extracting incident information..."):
        incident = extract_incident(incident_text)

    st.subheader("📋 Structured Incident Information")
    st.json(incident)

    st.subheader("📚 Retrieved Safety Information")

    rag_query = f"""
Emergency type: {incident.get("hazard")}
Location: {incident.get("location")}
Severity: {incident.get("severity")}
Blocked exit: {incident.get("blocked_exit")}

What safety procedures and evacuation guidance should be followed for this emergency?
"""

    with st.spinner("Retrieving relevant safety knowledge..."):
        safety_documents = retrieve_safety_information(rag_query)

    for i, document in enumerate(safety_documents, start=1):
        with st.expander(f"Safety Source {i}"):
            st.write(document.page_content)
            if document.metadata:
                st.caption(
                    f"Source: {document.metadata.get('source', 'Unknown')}"
                )

    incident_graph = apply_incident_to_graph(
        building_graph,
        incident.get("location", start_location),
        incident.get("severity", "medium"),
        incident.get("blocked_exit", "N/A")
    )

    route, route_cost, selected_exit = find_safest_route(
        incident_graph,
        start_location,
        available_exits
    )

    st.subheader("🗺️ Dynamic Evacuation Route")

    if route:
        st.success(f"Recommended Exit: {selected_exit}")
        st.write(" → ".join(route))
        st.metric("Route Cost", round(route_cost, 2))
        st.pyplot(draw_building_map(incident_graph, route))
    else:
        st.error("No feasible evacuation route is currently available.")
        st.pyplot(draw_building_map(incident_graph))

    st.subheader("🤖 CrisisLens Decision")

    if route:
        severity = incident.get("severity", "medium")
        hazard = incident.get("hazard", "unknown")
        blocked_exit = incident.get("blocked_exit", "N/A")

        st.markdown(f"""
### 🚨 Emergency Assessment

**Hazard:** {hazard}

**Severity:** {severity}

**Blocked Exit:** {blocked_exit}

**Recommended Evacuation Route:**

`{" → ".join(route)}`

**Recommended Exit:** {selected_exit}

The route was selected using graph-based optimization that considers
both travel distance and dynamically assigned hazard risk.

The retrieved safety information should be followed together with
instructions from emergency authorities.
""")
    else:
        st.error(
            "CrisisLens could not identify a feasible evacuation route. "
            "Emergency authorities should be contacted immediately."
        )

st.divider()
st.caption(
    "CrisisLens — AI-Powered Emergency Response and Dynamic Evacuation System | Research PoC"
)
