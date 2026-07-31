# Configuration and Prompts

KEYWORD_EXTRACTION_PROMPT = """
From the following text, extract the top 20 buzzwords related to the field:
{text}
Return as a comma-separated list.
"""

COMBINATION_GENERATION_PROMPT = """
Using these keywords: {keywords}, generate 50 unique combinations of 2-5 terms that represent potential research topics.
Return as a numbered list.
"""

RANKING_PROMPT = """
Rank the following 50 research topic combinations.
For each, consider current research trends, existing academic papers, and technological feasibility.
Return the top 5, with a brief justification for each rank based on this analysis:
{combinations}
"""

PROPOSAL_GENERATION_PROMPT = """
Generate a concrete, detailed research proposal for the topic: {topic}.
Include: Title, Abstract, Detailed Problem Statement, Proposed Methodology (with mathematical formulation), and Expected Impact.
"""

WORKFLOW_PLANNING_PROMPT = """
Given the research topic: {topic}.
Available skills:
{skills_context}

Generate a JSON research workflow. 
YOU MUST include the following mandatory phases in order:
1. 'Literature and Novelty Analysis' (identify gaps).
2. 'Research Proposal Construction' (define methodology).
3. 'Implementation and Experimentation' (code/simulation).
4. 'Adversarial Verification' (rigorous critique).
5. 'Formal Paper Drafting' (synthesis).

For each step, assign a skill and clearly define the success criteria (the 'goal'). 
IF NO SUITABLE SKILL EXISTS, propose a new skill filename (ending in .md) and describe it.

Return ONLY a JSON array of objects:
[{{ "step": "Step Name", "skill": "exact_filename.md", "description": "Specific task to perform", "goal": "What success looks like for this step" }}]
"""

LITERATURE_RETRIEVAL_PROMPT = """
Search for recent literature related to: {topic}.
Summarize the key findings and methodologies of the top 3 most relevant papers.
This will be used as a context for novelty checking.
"""

NOVELTY_CHECK_PROMPT = """
Context literature findings:
{literature_context}

Proposed Research:
{proposal}

Is the proposed research novel compared to the context literature?
Return 'VALID' if it is novel and valuable, or 'INVALID' followed by a detailed explanation why it is redundant or not novel.
"""

PAPER_DRAFTING_PROMPT = """
Write a comprehensive, professional-grade academic research paper in LaTeX format.

Requirements:
- Project Structure: Create a main.tex file that uses \\input{{}} to include modular files for: 
    abstract.tex, introduction.tex, literature_review.tex, methodology.tex, results.tex, discussion.tex, conclusion.tex, references.bib.
- Formatting: Use the standard 'article' document class (or similar professional template). Ensure all LaTeX commands for math, figures, and citations are syntactically correct.
- Depth & Length: This is a discovery of high magnitude. Ensure the content is exhaustive, professional, and structured for a submission to a top-tier journal. Total content should be substantial.
- Context:
{research_context}

Return the content for each file separately in the following format:
FILE: [filename.tex]
[Content of the file]
"""

PLANNING_AND_CRITIQUE_PROMPT = """
You are a research planner. Your goal is to design a high-impact, novel, and feasible research workflow.

Research Topic: {topic}
Skills Context: {skills_context}

Iterative Process:
1. Propose a JSON research workflow with the mandatory phases:
    - Literature and Novelty Analysis
    - Research Proposal Construction
    - Implementation and Experimentation
    - Adversarial Verification
    - Formal Paper Drafting

2. Assess your own proposal against the 'Novelty Assessment Skill' (provided in skills_context).

3. If you find the proposal is not novel or not feasible, refine the proposal and repeat the assessment until you achieve a high score, then output the final JSON.

Return ONLY the final JSON array of objects:
[{{ "step": "Step Name", "skill": "exact_filename.md", "description": "Specific task to perform", "goal": "What success looks like for this step" }}]
"""

TOPIC_SELECTION_PROMPT = """
Given the general research field: {field}.
Propose 5 specific, high-impact, novel research topics within this field.

For each topic, provide a brief title and a 1-sentence description.

Return ONLY JSON:
{{
  "research_topics": [
    {{ "number": 1, "title": "...", "description": "..." }},
    ...
  ]
}}
"""
