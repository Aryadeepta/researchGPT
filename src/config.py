
# Configuration and Prompts

FAST_QUEUE = ["models/gemini-3.1-flash-lite", "models/gemini-3.5-flash"]
SMART_QUEUE = ["models/gemini-3.1-flash-lite", "models/gemini-3.5-flash", "models/gemini-2.0-flash"]

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

STYLE_GUIDE_PROMPT = """
Research the common academic style, structure, sections, and length for a high-impact research paper on: {topic}.
Return ONLY JSON:
{{
  "sections": ["abstract", "introduction", "literature_review", "methodology", "results", "discussion", "conclusion"],
  "latex_class": "article",
  "recommended_length_pages": 15
}}
"""

MAIN_TEX_PROMPT = """
Create a `main.tex` file content for a paper titled '{topic}'.
Structure:
- Use \\documentclass{{{latex_class}}}
- Include essential packages for math and academic formatting (amsmath, amssymb, graphicx, etc.).
- Use \\input{{sections/filename}} for each of these sections: {sections}.
- Do NOT include \\section{{}} headers in this file. Only the document structure and input commands.
Return ONLY the LaTeX code for main.tex.
"""

SECTION_DRAFTING_PROMPT = """
Draft the content for the {section_name} section of a paper titled '{topic}'.

Requirements:
- Structure: Start with \\section{{{section_name_title}}}. Do NOT include documentclass or begin/end document.
- Depth & Length: This is a high-magnitude discovery. Be exhaustive. 
    - Methodology/Results/Literature Review: Aim for 1500+ words per section. Provide detailed, rigorous mathematical derivations, comprehensive analysis, and extensive context. Do not generalize or summarize.
- Content: Use LaTeX for all math ($...$, $$...$$). Provide robust argumentation.
- Claim Provenance: You MUST verify all research claims against the provided Evidence Ledger.
    - If a claim exists in the ledger and its status is 'UNVERIFIED', you MUST use the provided 'allowed_paper_language' for that claim. DO NOT state it as a result.
    - If you include a claim not in the ledger, mark it clearly as HYPOTHESIS or ANLYTICAL_ESTIMATE.
- Evidence Ledger:
{evidence_ledger}
- Context:
{research_context}

Return ONLY the LaTeX content for this section.
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

For each step, include:
- 'step': Name
- 'skill': Skill filename
- 'description': Detailed task
- 'goal': Success objective
- 'expected_artifacts': List of file paths this step MUST produce (e.g., ['novelty_gap_analysis.json', 'math_formulation.md']).
- 'implementation_instruction': 'CRITICAL: Your Python code block MUST include file writing operations for EVERY file listed in expected_artifacts using: with open(filepath, "w") as f: f.write(content). Do not just print or log the content. If you fail to create the file, the pipeline will block.'

Return ONLY the final JSON array of objects:
[{"step": "...", "skill": "...", "description": "...", "goal": "...", "expected_artifacts": ["..."], "implementation_instruction": "..." }, ...]
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
