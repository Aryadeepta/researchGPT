import argparse
import os
from src.agent import ResearchAgent
from src.rag import ProjectRAG

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir", help="Path to the project directory")
    args = parser.parse_args()

    if not os.path.exists(args.project_dir):
        print(f"Error: Project directory '{args.project_dir}' not found.")
        return

    # Initialize adversary agent for QA
    adversary = ResearchAgent("You are an adversarial reviewer. Answer questions based on research context.")
    
    # Initialize RAG
    rag = ProjectRAG(args.project_dir)
    
    print(f"Loaded research context from {args.project_dir}")
    print("You can now ask questions about this research. Type 'exit' to quit.")
    while True:
        question = input("\nQuestion: ")
        if question.lower() == 'exit': break
        print("Answer:", rag.query(adversary, question))

if __name__ == "__main__":
    main()
