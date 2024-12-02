# Content Engine

## Overview
The **Content Engine** is an interactive application for analyzing and comparing multiple PDF documents. It uses Retrieval-Augmented Generation (RAG) to retrieve and generate insights based on uploaded PDFs. This project includes:
- Text extraction from PDFs
- Vector-based semantic search
- Local Large Language Model (LLM) for contextual insights
- Streamlit-based chatbot interface

## Features
- Upload and process multiple PDF documents.
- Extract text and generate embeddings for semantic analysis.
- Compare documents and answer user queries interactively.
- Maintain data privacy by running all computations locally.

## Technologies Used
- **Streamlit**: Frontend for user interaction.
- **PyPDF2**: For extracting text from PDF files.
- **SentenceTransformers**: To generate embeddings.
- **ChromaDB**: For managing vector-based retrieval.
- **Transformers**: For running a local LLM.

## Installation

### Prerequisites
Ensure you have Python 3.8 or above installed on your system.

### Install Dependencies
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/content-engine.git
   cd content-engine
