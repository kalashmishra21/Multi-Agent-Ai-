"""
Multi-Agent AI Study Planner - Flask Web Application
=====================================================
Web interface for the 4-agent LangChain + LangGraph study planner.
Streams agent progress in real-time to the browser using SSE.

Author  : AI Study Planner Project
Version : 3.0.0
"""

import os
import json
import sys
from typing import Optional, Generator

from flask import Flask, render_template, request, Response, jsonify
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------

load_dotenv()
app = Flask(__name__)


# ---------------------------------------------------------------------------
# LLM Initializer
# ---------------------------------------------------------------------------

def get_llm() -> ChatGoogleGenerativeAI:
    """
    Initialize and return the Google Gemini LLM instance.

    Returns:
        ChatGoogleGenerativeAI: Configured Gemini LLM instance.

    Raises:
        RuntimeError: If GOOGLE_API_KEY is not set.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.strip() == "":
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. "
            "Please add it to your .env file. "
            "Get a free key at: https://aistudio.google.com/app/apikey"
        )
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        google_api_key=api_key,
        convert_system_message_to_human=True,
    )


# ---------------------------------------------------------------------------
# Shared State
# ---------------------------------------------------------------------------

class StudyPlanState(TypedDict):
    """Shared state passed between all agents in the LangGraph pipeline."""
    user_input: dict
    requirements: str
    topics: str
    study_plan: str
    final_output: str
    error: Optional[str]


# ---------------------------------------------------------------------------
# Agent Functions
# ---------------------------------------------------------------------------

def requirement_analyzer_agent(state: StudyPlanState) -> StudyPlanState:
    """Agent 1: Analyzes user input and extracts structured requirements."""
    try:
        llm = get_llm()
        user_data = state["user_input"]

        system_prompt = (
            "You are an expert educational consultant and requirement analyst. "
            "Your job is to analyze a student's learning request and extract "
            "structured, actionable requirements. Be precise and thorough."
        )

        human_prompt = f"""
Analyze the following student learning request and extract structured requirements:

Subject       : {user_data.get('subject', 'Not specified')}
Goal          : {user_data.get('goal', 'Not specified')}
Available Time: {user_data.get('hours_per_day', 'Not specified')} hours per day
Deadline      : {user_data.get('deadline', 'Not specified')}
Learning Style: {user_data.get('learning_style', 'Not specified')}

Please provide a structured analysis with the following sections:
1. LEARNING OBJECTIVE     - Clear, measurable goal
2. TIME CONSTRAINTS       - Total available hours, daily commitment
3. SKILL LEVEL ASSESSMENT - Assumed current level based on goal
4. LEARNING STYLE NOTES   - How to tailor content delivery
5. KEY CONSTRAINTS        - Any limitations or special considerations
6. SUCCESS CRITERIA       - How to measure completion

Be specific, realistic, and actionable.
"""
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        response = llm.invoke(messages)
        return {**state, "requirements": response.content.strip(), "error": None}

    except RuntimeError:
        raise
    except Exception as e:
        return {**state, "requirements": "", "error": f"Agent 1 failed: {str(e)}"}


def research_agent(state: StudyPlanState) -> StudyPlanState:
    """Agent 2: Breaks subject into ordered topics and curriculum."""
    if state.get("error"):
        return state
    try:
        llm = get_llm()
        user_data = state["user_input"]
        requirements = state["requirements"]

        system_prompt = (
            "You are a curriculum design expert and subject matter specialist. "
            "Break down any subject into a well-structured, progressive learning curriculum."
        )

        human_prompt = f"""
Based on the following requirements, create a comprehensive topic breakdown for:
Subject: {user_data.get('subject', 'the requested subject')}

REQUIREMENTS ANALYSIS:
{requirements}

Please provide:

1. PREREQUISITES (if any)
   - List what the student should already know

2. CORE TOPICS (organized in learning order)
   - Topic 1: [Name] | Difficulty: [Beginner/Intermediate/Advanced] | Est. Hours: [X]
   - Topic 2: [Name] | Difficulty: [...] | Est. Hours: [X]
   (Continue for all major topics)

3. PRACTICAL SKILLS
   - Hands-on projects and exercises for each major topic

4. MILESTONE CHECKPOINTS
   - Key points where the student should self-assess

5. RECOMMENDED RESOURCES
   - Books, courses, websites, or tools for each topic cluster

6. TOPIC DEPENDENCIES
   - Which topics must be completed before others

Be comprehensive. Include at least 8-12 core topics.
"""
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        response = llm.invoke(messages)
        return {**state, "topics": response.content.strip(), "error": None}

    except RuntimeError:
        raise
    except Exception as e:
        return {**state, "topics": "", "error": f"Agent 2 failed: {str(e)}"}


def planner_agent(state: StudyPlanState) -> StudyPlanState:
    """Agent 3: Creates a detailed day-wise study schedule."""
    if state.get("error"):
        return state
    try:
        llm = get_llm()
        user_data = state["user_input"]
        requirements = state["requirements"]
        topics = state["topics"]

        system_prompt = (
            "You are an expert academic planner and productivity coach. "
            "Create realistic, detailed, day-by-day study schedules that are "
            "achievable, well-paced, and include revision and practice."
        )

        human_prompt = f"""
Create a complete day-wise study plan based on the following:

STUDENT PROFILE:
- Subject       : {user_data.get('subject', 'Not specified')}
- Goal          : {user_data.get('goal', 'Not specified')}
- Hours Per Day : {user_data.get('hours_per_day', 'Not specified')} hours
- Deadline      : {user_data.get('deadline', 'Not specified')}
- Learning Style: {user_data.get('learning_style', 'Not specified')}

REQUIREMENTS ANALYSIS:
{requirements}

TOPIC BREAKDOWN:
{topics}

Create a COMPLETE day-wise plan with this structure:

WEEK-BY-WEEK OVERVIEW
- Week 1: [Theme/Focus]
- Week 2: [Theme/Focus]
(Continue for all weeks)

DETAILED DAY-WISE SCHEDULE
For each day provide:
Day X (Week Y):
  - Topic: [Specific topic]
  - Activities: [What to study/do]
  - Duration: [Hours breakdown]
  - Resources: [Specific resource to use]
  - Practice Task: [Hands-on exercise]
  - Goal: [What to achieve by end of day]

REVISION SCHEDULE
- Weekly revision days and what to revise

PRACTICE PROJECTS
- Mini-projects and final capstone project

DAILY ROUTINE TEMPLATE
- Suggested time blocks for the {user_data.get('hours_per_day', '2')} hours

Make the plan complete for the entire {user_data.get('deadline', 'given')} period.
"""
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        response = llm.invoke(messages)
        return {**state, "study_plan": response.content.strip(), "error": None}

    except RuntimeError:
        raise
    except Exception as e:
        return {**state, "study_plan": "", "error": f"Agent 3 failed: {str(e)}"}


def reviewer_agent(state: StudyPlanState) -> StudyPlanState:
    """Agent 4: Reviews, improves, and finalizes the study plan."""
    if state.get("error"):
        return state
    try:
        llm = get_llm()
        user_data = state["user_input"]
        requirements = state["requirements"]
        study_plan = state["study_plan"]

        requirements_preview = requirements[:800] if len(requirements) > 800 else requirements

        system_prompt = (
            "You are a senior academic advisor and quality assurance expert. "
            "Critically review study plans, identify gaps, improve weak sections, "
            "and produce a final polished student-ready study roadmap."
        )

        human_prompt = f"""
Review the following study plan and produce a final, optimized version.

STUDENT PROFILE:
- Subject       : {user_data.get('subject', 'Not specified')}
- Goal          : {user_data.get('goal', 'Not specified')}
- Hours Per Day : {user_data.get('hours_per_day', 'Not specified')} hours
- Deadline      : {user_data.get('deadline', 'Not specified')}
- Learning Style: {user_data.get('learning_style', 'Not specified')}

ORIGINAL STUDY PLAN:
{study_plan}

Please:
1. REVIEW for feasibility - Is the pace realistic?
2. CHECK completeness - Are all important topics covered?
3. IDENTIFY weak sections - What needs improvement?
4. IMPROVE the plan - Fix any issues found
5. ADD motivational tips and study strategies
6. PRODUCE the final optimized plan

Output the FINAL PLAN using this structure:

===================================================================
           PERSONALIZED STUDY PLAN - FINAL VERSION
===================================================================

REQUIREMENTS SUMMARY:
{requirements_preview}

-------------------------------------------------------------------
LEARNING OBJECTIVES:
[List clear, measurable objectives]

-------------------------------------------------------------------
TOPIC ROADMAP:
[Organized topics in learning order]

-------------------------------------------------------------------
DAY-WISE STUDY SCHEDULE:
[Complete day-by-day plan]

-------------------------------------------------------------------
REVISION STRATEGY:
[How and when to revise]

-------------------------------------------------------------------
PRACTICE PROJECTS:
[Projects and exercises]

-------------------------------------------------------------------
PROGRESS MILESTONES:
[Weekly checkpoints]

-------------------------------------------------------------------
STUDY TIPS AND STRATEGIES:
[Personalized tips based on the student's learning style]

-------------------------------------------------------------------
IMPORTANT REMINDERS:
[Key advice for success]

-------------------------------------------------------------------
SUCCESS CRITERIA:
[How the student will know they have achieved their goal]

===================================================================
"""
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        response = llm.invoke(messages)
        return {**state, "final_output": response.content.strip(), "error": None}

    except RuntimeError:
        raise
    except Exception as e:
        return {**state, "final_output": "", "error": f"Agent 4 failed: {str(e)}"}


# ---------------------------------------------------------------------------
# LangGraph Pipeline
# ---------------------------------------------------------------------------

def build_graph():
    """Build and compile the LangGraph StateGraph pipeline."""
    graph = StateGraph(StudyPlanState)
    graph.add_node("requirement_analyzer", requirement_analyzer_agent)
    graph.add_node("research_agent", research_agent)
    graph.add_node("planner_agent", planner_agent)
    graph.add_node("reviewer_agent", reviewer_agent)
    graph.set_entry_point("requirement_analyzer")
    graph.add_edge("requirement_analyzer", "research_agent")
    graph.add_edge("research_agent", "planner_agent")
    graph.add_edge("planner_agent", "reviewer_agent")
    graph.add_edge("reviewer_agent", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# SSE Stream Generator
# ---------------------------------------------------------------------------

def run_pipeline_stream(user_input: dict) -> Generator[str, None, None]:
    """
    Run the 4-agent pipeline and yield SSE events for each step.
    This allows the browser to show real-time agent progress.

    Args:
        user_input: Dictionary of user-provided study details.

    Yields:
        SSE-formatted strings with agent status and final output.
    """

    def sse(event: str, data: dict) -> str:
        """Format a Server-Sent Event message."""
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    try:
        # Validate API key first
        get_llm()
    except RuntimeError as e:
        yield sse("error", {"message": str(e)})
        return

    # --- Agent 1 ---
    yield sse("agent_start", {"agent": 1, "name": "Requirement Analyzer", "message": "Analyzing your requirements..."})

    state: StudyPlanState = {
        "user_input": user_input,
        "requirements": "",
        "topics": "",
        "study_plan": "",
        "final_output": "",
        "error": None,
    }

    try:
        state = requirement_analyzer_agent(state)
        if state.get("error"):
            yield sse("error", {"message": state["error"]})
            return
        yield sse("agent_done", {"agent": 1, "name": "Requirement Analyzer", "output": state["requirements"]})

        # --- Agent 2 ---
        yield sse("agent_start", {"agent": 2, "name": "Research Agent", "message": "Researching topics and curriculum..."})
        state = research_agent(state)
        if state.get("error"):
            yield sse("error", {"message": state["error"]})
            return
        yield sse("agent_done", {"agent": 2, "name": "Research Agent", "output": state["topics"]})

        # --- Agent 3 ---
        yield sse("agent_start", {"agent": 3, "name": "Planner Agent", "message": "Creating your day-wise study plan..."})
        state = planner_agent(state)
        if state.get("error"):
            yield sse("error", {"message": state["error"]})
            return
        yield sse("agent_done", {"agent": 3, "name": "Planner Agent", "output": state["study_plan"]})

        # --- Agent 4 ---
        yield sse("agent_start", {"agent": 4, "name": "Reviewer Agent", "message": "Reviewing and finalizing your plan..."})
        state = reviewer_agent(state)
        if state.get("error"):
            yield sse("error", {"message": state["error"]})
            return
        yield sse("agent_done", {"agent": 4, "name": "Reviewer Agent", "output": state["final_output"]})

        # --- Final ---
        yield sse("complete", {"final_output": state["final_output"]})

    except RuntimeError as e:
        yield sse("error", {"message": str(e)})
    except Exception as e:
        yield sse("error", {"message": f"Unexpected error: {str(e)}"})


# ---------------------------------------------------------------------------
# Flask Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the main web UI."""
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    """
    Accept form data and stream agent progress back to the browser via SSE.

    Expects JSON body with: subject, goal, hours_per_day, deadline, learning_style.
    Returns a text/event-stream response.
    """
    data = request.get_json()

    # Server-side validation
    subject = (data.get("subject") or "").strip()
    goal = (data.get("goal") or "").strip()
    hours_per_day = (data.get("hours_per_day") or "").strip()
    deadline = (data.get("deadline") or "").strip()
    learning_style = (data.get("learning_style") or "").strip()

    if not all([subject, goal, hours_per_day, deadline, learning_style]):
        return jsonify({"error": "All fields are required."}), 400

    try:
        hours = float(hours_per_day)
        if hours <= 0 or hours > 24:
            return jsonify({"error": "Hours per day must be between 0.5 and 24."}), 400
    except ValueError:
        return jsonify({"error": "Hours per day must be a valid number."}), 400

    user_input = {
        "subject": subject,
        "goal": goal,
        "hours_per_day": hours_per_day,
        "deadline": deadline,
        "learning_style": learning_style,
    }

    return Response(
        run_pipeline_stream(user_input),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
