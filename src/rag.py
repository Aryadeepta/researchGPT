import os
import re
import json

class ProjectRAG:
    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.knowledge_base = ""
        self._index_project()

    def _index_project(self):
        """Simple RAG indexer: collects all md and py files in the project dir."""
        for root, _, files in os.walk(self.project_dir):
            for file in files:
                if file.endswith((".md", ".py")):
                    with open(os.path.join(root, file), 'r') as f:
                        self.knowledge_base += f"\n--- File: {file} ---\n{f.read()}\n"

    def query(self, agent, question):
        """Uses the agent to answer questions based on the indexed knowledge base."""
        prompt = f"Using the following project research context, answer the user question.\n\nContext:\n{self.knowledge_base}\n\nQuestion: {question}"
        return agent.chat(prompt)

def summarize_project(agent, context):
    """Generates a high-level summary of the research."""
    prompt = f"Provide a concise summary of the research conducted based on this context: {context}"
    return agent.chat(prompt)
