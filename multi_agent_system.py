"""
Multi-Agent AI Study Planner System
=====================================
Uses LangChain + LangGraph to build a 4-agent pipeline that generates
a personalized, day-wise study roadmap based on user input.

LLM Backend : Google Gemini (via langchain-google-genai)
Framework   : LangChain + LangGraph

Agents:
    1. Requirement Analyzer  - Extracts and normalizes user input
    2. Research Agent        - Breaks subject into ordered topics
    3. Planner Agent         - Creates day-wise schedule
    4. Reviewer Agent        - Reviews, polishes, and finalizes the plan

Author  : AI Study Planner Project
Version : 2.0.0
"""

import os
import sys
from typing import Optional

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------------------------

load_dotenv()


def get_llm() -> ChatGoogleGenerativeAI:
    """
    Initialize and return the Google Gemini LLM instance via LangChain.

    Reads GOOGLE_API_KEY from environment. Raises a clear RuntimeError
    if the key is missing so the user knows exactly what to fix.

    Returns:
        ChatGoogleGenerativeAI: Configured Gemini LLM instance.

    Raises:
        RuntimeError: If GOOGLE_API_KEY is not set.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.strip() == "":
        raise RuntimeError(
            "\n[ERROR] GOOGLE_API_KEY is not set.\n"
            "Please create a .env file and add:\n"
            "  GOOGLE_API_KEY=your_google_gemini_api_key_here\n"
            "Get your free API key at: https://aistudio.google.com/app/apikey\n"
            "See .env.example for reference.\n"
        )
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",   # Fast, free-tier friendly Gemini model
        temperature=0.7,
        google_api_key=api_key,
        convert_system_message_to_human=True,  # Gemini requires this for system msgs
    )


# ---------------------------------------------------------------------------
# Shared State Definition (TypedDict)
# ---------------------------------------------------------------------------

class StudyPlanState(TypedDict):
    """
    Shared state passed between all agents in the LangGraph pipeline.

    Fields:
        user_input   : Raw dictionary of user-provided inputs.
        requirements : Structured requirements extracted by Agent 1.
        topics       : Ordered topic breakdown produced by Agent 2.
        study_plan   : Day-wise schedule created by Agent 3.
        final_output : Polished, reviewer-approved plan from Agent 4.
        error        : Optional error message if any agent fails.
    """
    user_input: dict
    requirements: str
    topics: str
    study_plan: str
    final_output: str
    error: Optional[str]


# ---------------------------------------------------------------------------
# Agent 1 - Requirement Analyzer
# ---------------------------------------------------------------------------

def requirement_analyzer_agent(state: StudyPlanState) -> StudyPlanState:
    """
    Agent 1: Requirement Analyzer

    Reads raw user input, understands the learning goal, extracts key
    constraints (time, deadline, style), and returns a structured
    requirements summary that downstream agents can rely on.

    Args:
        state: Current pipeline state containing user_input.

    Returns:
        Updated state with 'requirements' field populated.
    """
    print("\n[Agent 1] Analyzing requirements...")

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

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        response = llm.invoke(messages)
        requirements = response.content.strip()

        print("[Agent 1] Requirements analyzed successfully.")
        return {**state, "requirements": requirements, "error": None}

    except RuntimeError:
        # API key missing - propagate immediately
        raise
    except Exception as e:
        error_msg = f"[Agent 1 ERROR] Requirement analysis failed: {str(e)}"
        print(error_msg)
        return {**state, "requirements": "", "error": error_msg}


# ---------------------------------------------------------------------------
# Agent 2 - Research Agent
# ---------------------------------------------------------------------------

def research_agent(state: StudyPlanState) -> StudyPlanState:
    """
    Agent 2: Research Agent

    Takes the structured requirements and breaks the subject into an
    ordered list of topics, identifies prerequisites, and organizes
    the learning sequence from fundamentals to advanced concepts.

    Args:
        state: Current pipeline state with 'requirements' populated.

    Returns:
        Updated state with 'topics' field populated.
    """
    print("\n[Agent 2] Researching topics and curriculum...")

    # If a previous agent encountered an error, skip gracefully
    if state.get("error"):
        print("[Agent 2] Skipping due to upstream error.")
        return state

    try:
        llm = get_llm()
        user_data = state["user_input"]
        requirements = state["requirements"]

        system_prompt = (
            "You are a curriculum design expert and subject matter specialist. "
            "Your job is to break down any subject into a well-structured, "
            "progressive learning curriculum. Organize topics from foundational "
            "to advanced, ensuring logical progression."
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

Be comprehensive. Include at least 8-12 core topics for a complete curriculum.
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        response = llm.invoke(messages)
        topics = response.content.strip()

        print("[Agent 2] Topic research completed successfully.")
        return {**state, "topics": topics, "error": None}

    except RuntimeError:
        raise
    except Exception as e:
        error_msg = f"[Agent 2 ERROR] Research failed: {str(e)}"
        print(error_msg)
        return {**state, "topics": "", "error": error_msg}


# ---------------------------------------------------------------------------
# Agent 3 - Planner Agent
# ---------------------------------------------------------------------------

def planner_agent(state: StudyPlanState) -> StudyPlanState:
    """
    Agent 3: Planner Agent

    Combines the requirements and topic breakdown to create a detailed,
    day-wise study schedule. Allocates study hours, adds revision
    sessions, and includes practice tasks and mini-projects.

    Args:
        state: Current pipeline state with 'requirements' and 'topics'.

    Returns:
        Updated state with 'study_plan' field populated.
    """
    print("\n[Agent 3] Creating personalized day-wise study plan...")

    if state.get("error"):
        print("[Agent 3] Skipping due to upstream error.")
        return state

    try:
        llm = get_llm()
        user_data = state["user_input"]
        requirements = state["requirements"]
        topics = state["topics"]

        system_prompt = (
            "You are an expert academic planner and productivity coach. "
            "Your job is to create realistic, detailed, day-by-day study schedules "
            "that are achievable, well-paced, and include revision and practice. "
            "Always account for human learning curves and avoid burnout."
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
- Weekly revision days
- What to revise and how

PRACTICE PROJECTS
- Mini-projects scheduled throughout the plan
- Final capstone project

DAILY ROUTINE TEMPLATE
- Suggested time blocks for the {user_data.get('hours_per_day', '2')} hours

PROGRESS TRACKING
- Weekly milestones
- Self-assessment checkpoints

Make the plan complete for the entire {user_data.get('deadline', 'given')} period.
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        response = llm.invoke(messages)
        study_plan = response.content.strip()

        print("[Agent 3] Study plan created successfully.")
        return {**state, "study_plan": study_plan, "error": None}

    except RuntimeError:
        raise
    except Exception as e:
        error_msg = f"[Agent 3 ERROR] Planning failed: {str(e)}"
        print(error_msg)
        return {**state, "study_plan": "", "error": error_msg}


# ---------------------------------------------------------------------------
# Agent 4 - Reviewer Agent
# ---------------------------------------------------------------------------

def reviewer_agent(state: StudyPlanState) -> StudyPlanState:
    """
    Agent 4: Reviewer Agent

    Reviews the generated study plan for feasibility, completeness,
    and quality. Identifies weak sections, improves them, and returns
    a final polished, optimized study plan ready for the student.

    Args:
        state: Current pipeline state with all previous agent outputs.

    Returns:
        Updated state with 'final_output' field populated.
    """
    print("\n[Agent 4] Reviewing and finalizing the study plan...")

    if state.get("error"):
        print("[Agent 4] Skipping due to upstream error.")
        return state

    try:
        llm = get_llm()
        user_data = state["user_input"]
        requirements = state["requirements"]
        study_plan = state["study_plan"]

        system_prompt = (
            "You are a senior academic advisor and quality assurance expert. "
            "Your job is to critically review study plans, identify gaps or "
            "unrealistic expectations, improve weak sections, and produce a "
            "final, polished, student-ready study roadmap. Be constructive, "
            "encouraging, and precise."
        )

        # Truncate requirements to avoid exceeding context limits
        requirements_preview = requirements[:800] if len(requirements) > 800 else requirements

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

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        response = llm.invoke(messages)
        final_output = response.content.strip()

        print("[Agent 4] Review and optimization completed.")
        return {**state, "final_output": final_output, "error": None}

    except RuntimeError:
        raise
    except Exception as e:
        error_msg = f"[Agent 4 ERROR] Review failed: {str(e)}"
        print(error_msg)
        return {**state, "final_output": "", "error": error_msg}


# ---------------------------------------------------------------------------
# LangGraph Pipeline Builder
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Build and compile the LangGraph StateGraph pipeline.

    Defines nodes for each agent and connects them with directed edges
    in the order: Analyzer -> Research -> Planner -> Reviewer -> END.

    Returns:
        Compiled LangGraph application ready to invoke.
    """
    # Initialize the graph with our shared state schema
    graph = StateGraph(StudyPlanState)

    # Register each agent as a named node
    graph.add_node("requirement_analyzer", requirement_analyzer_agent)
    graph.add_node("research_agent", research_agent)
    graph.add_node("planner_agent", planner_agent)
    graph.add_node("reviewer_agent", reviewer_agent)

    # Define the execution flow with directed edges
    graph.set_entry_point("requirement_analyzer")
    graph.add_edge("requirement_analyzer", "research_agent")
    graph.add_edge("research_agent", "planner_agent")
    graph.add_edge("planner_agent", "reviewer_agent")
    graph.add_edge("reviewer_agent", END)

    # Compile and return the executable graph
    return graph.compile()


# ---------------------------------------------------------------------------
# Input Collection Helpers
# ---------------------------------------------------------------------------

def get_non_empty_input(prompt: str) -> str:
    """
    Prompt the user for input and validate it is not empty.

    Keeps asking until a non-empty, non-whitespace value is provided.

    Args:
        prompt: The prompt string to display to the user.

    Returns:
        A non-empty string entered by the user.
    """
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("  [!] This field cannot be empty. Please enter a value.")


def get_valid_hours() -> str:
    """
    Prompt the user for daily study hours and validate the input.

    Accepts values between 0.5 and 24. Keeps asking until valid input
    is provided.

    Returns:
        A string representation of valid hours per day.
    """
    while True:
        raw = input("Enter available study hours per day (e.g., 2): ").strip()
        if not raw:
            print("  [!] Hours cannot be empty.")
            continue
        try:
            hours = float(raw)
            if hours <= 0:
                print("  [!] Hours must be greater than 0.")
            elif hours > 24:
                print("  [!] Hours cannot exceed 24 per day.")
            else:
                return str(hours)
        except ValueError:
            print("  [!] Please enter a valid number (e.g., 2 or 1.5).")


def get_valid_deadline() -> str:
    """
    Prompt the user for a deadline and validate it is meaningful.

    Accepts any non-empty string (e.g., '30 days', '3 months', 'June 2025').
    Ensures the input is at least 3 characters to avoid garbage input.

    Returns:
        A valid deadline string entered by the user.
    """
    while True:
        raw = input("Enter deadline (e.g., '30 days', '3 months', 'June 2025'): ").strip()
        if not raw:
            print("  [!] Deadline cannot be empty.")
            continue
        if len(raw) < 3:
            print("  [!] Please provide a more descriptive deadline.")
            continue
        return raw


def collect_user_input() -> dict:
    """
    Interactively collect all required inputs from the user via terminal.

    Validates each field before accepting it. Returns a clean dictionary
    of user inputs ready to be passed into the agent pipeline.

    Returns:
        dict: Contains subject, goal, hours_per_day, deadline, learning_style.
    """
    print("\n" + "=" * 60)
    print("       AI STUDY PLANNER - Multi-Agent System")
    print("         Powered by Google Gemini + LangGraph")
    print("=" * 60)
    print("Please provide your study details below.\n")

    subject = get_non_empty_input("Enter subject (e.g., Machine Learning): ")
    goal = get_non_empty_input("Enter your goal (e.g., Get a job as ML engineer): ")
    hours_per_day = get_valid_hours()
    deadline = get_valid_deadline()

    print("Learning styles: visual / auditory / reading / kinesthetic / mixed")
    learning_style = get_non_empty_input("Enter your learning style: ")

    return {
        "subject": subject,
        "goal": goal,
        "hours_per_day": hours_per_day,
        "deadline": deadline,
        "learning_style": learning_style,
    }


# ---------------------------------------------------------------------------
# Output Formatter
# ---------------------------------------------------------------------------

def display_final_output(state: StudyPlanState) -> None:
    """
    Display the final study plan output in a clean, formatted layout.

    If an error occurred during the pipeline, displays the error message
    instead. Falls back to showing intermediate outputs if the final
    output is empty.

    Args:
        state: Final pipeline state after all agents have run.
    """
    print("\n" + "=" * 60)

    # Check for pipeline errors
    if state.get("error"):
        print("[ERROR] The pipeline encountered an error:")
        print(state["error"])
        print("=" * 60)
        return

    # Display final output if available
    if state.get("final_output"):
        print(state["final_output"])
    else:
        # Fallback: show whatever was generated by earlier agents
        print("STUDY PLAN SUMMARY")
        print("=" * 60)
        if state.get("requirements"):
            print("\nREQUIREMENTS:\n")
            print(state["requirements"])
        if state.get("topics"):
            print("\nTOPICS:\n")
            print(state["topics"])
        if state.get("study_plan"):
            print("\nSTUDY PLAN:\n")
            print(state["study_plan"])

    print("\n" + "=" * 60)
    print("Your personalized study plan is ready!")
    print("Save this output and start your learning journey.")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Main entry point for the AI Study Planner Multi-Agent System.

    Orchestrates the full pipeline:
        1. Collect user input interactively
        2. Build the LangGraph pipeline
        3. Initialize the shared state
        4. Invoke the graph (runs all 4 agents sequentially)
        5. Display the final formatted output

    Handles top-level exceptions including missing API keys and
    unexpected runtime errors with clear user-facing messages.
    """
    try:
        # Step 1: Collect user input
        user_input = collect_user_input()

        print("\n" + "=" * 60)
        print("  Starting Multi-Agent Pipeline...")
        print("  4 specialized AI agents will now process your request.")
        print("=" * 60)

        # Step 2: Build the LangGraph pipeline
        app = build_graph()

        # Step 3: Initialize shared state
        initial_state: StudyPlanState = {
            "user_input": user_input,
            "requirements": "",
            "topics": "",
            "study_plan": "",
            "final_output": "",
            "error": None,
        }

        # Step 4: Run the full agent pipeline
        final_state = app.invoke(initial_state)

        # Step 5: Display results
        display_final_output(final_state)

    except RuntimeError as e:
        # API key or configuration error
        print(str(e))
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n[INFO] Session interrupted by user. Goodbye!")
        sys.exit(0)

    except Exception as e:
        print(f"\n[FATAL ERROR] An unexpected error occurred: {str(e)}")
        print("Please check your configuration and try again.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Script Entry Guard
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
