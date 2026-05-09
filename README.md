# 🎓 AI Study Planner — Multi-Agent System

> A production-ready multi-agent AI system built with **LangChain**, **LangGraph**, and **Google Gemini** that generates personalized, day-wise study roadmaps using 4 specialized AI agents working in a coordinated pipeline.

---

## 📌 Objective

Students often struggle to create structured, realistic study plans. This system solves that by using 4 AI agents — each with a distinct role — to analyze your learning goal, research the subject, build a day-wise schedule, and review it for quality. The result is a complete, personalized study roadmap tailored to your time, deadline, and learning style.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  LANGGRAPH STATE MACHINE                    │
│                                                             │
│   ┌──────────────┐    ┌──────────────┐                      │
│   │  User Input  │───▶│   Agent 1    │                      │
│   │  (Terminal)  │    │  Requirement │                      │
│   └──────────────┘    │  Analyzer    │                      │
│                       └──────┬───────┘                      │
│                              │ requirements                 │
│                              ▼                              │
│                       ┌──────────────┐                      │
│                       │   Agent 2    │                      │
│                       │  Research    │                      │
│                       │  Agent       │                      │
│                       └──────┬───────┘                      │
│                              │ topics                       │
│                              ▼                              │
│                       ┌──────────────┐                      │
│                       │   Agent 3    │                      │
│                       │  Planner     │                      │
│                       │  Agent       │                      │
│                       └──────┬───────┘                      │
│                              │ study_plan                   │
│                              ▼                              │
│                       ┌──────────────┐                      │
│                       │   Agent 4    │                      │
│                       │  Reviewer    │                      │
│                       │  Agent       │                      │
│                       └──────┬───────┘                      │
│                              │ final_output                 │
│                              ▼                              │
│                           [ END ]                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 🤖 Agent Roles

| Agent | Name | Role | Output |
|-------|------|------|--------|
| 1 | **Requirement Analyzer** | Extracts learning objectives, constraints, skill level, and success criteria from raw user input | Structured requirements document |
| 2 | **Research Agent** | Breaks the subject into ordered topics, identifies prerequisites, and maps the learning sequence | Topic breakdown with difficulty and hours |
| 3 | **Planner Agent** | Creates a complete day-wise schedule with study hours, revision sessions, and practice tasks | Detailed day-by-day study plan |
| 4 | **Reviewer Agent** | Reviews the plan for feasibility, fills gaps, improves weak sections, and produces the final polished output | Final optimized study roadmap |

---

## 🔄 LangGraph Workflow

```
START
  │
  ▼
[requirement_analyzer]  ──▶  Extracts structured requirements
  │
  ▼
[research_agent]         ──▶  Builds topic curriculum
  │
  ▼
[planner_agent]          ──▶  Creates day-wise schedule
  │
  ▼
[reviewer_agent]         ──▶  Reviews and finalizes plan
  │
  ▼
END
```

**Shared State (TypedDict):**
```python
class StudyPlanState(TypedDict):
    user_input   : dict          # Raw user inputs
    requirements : str           # Agent 1 output
    topics       : str           # Agent 2 output
    study_plan   : str           # Agent 3 output
    final_output : str           # Agent 4 output
    error        : Optional[str] # Error propagation
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.9 or higher
- A Google Gemini API key ([get free key here](https://aistudio.google.com/app/apikey))

### Step 1 — Clone the repository
```bash
git clone https://github.com/yourusername/ai-study-planner.git
cd ai-study-planner
```

### Step 2 — Create a virtual environment
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Set up environment variables
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your Google Gemini API key
# GOOGLE_API_KEY=your_actual_key_here
# Get free key at: https://aistudio.google.com/app/apikey
```

---

## ▶️ How to Run

```bash
python multi_agent_system.py
```

The program will interactively ask for your details:

```
============================================================
       AI STUDY PLANNER - Multi-Agent System
============================================================
Please provide your study details below.

Enter subject (e.g., Machine Learning): 
Enter your goal (e.g., Get a job as ML engineer): 
Enter available study hours per day (e.g., 2): 
Enter deadline (e.g., '30 days', '3 months', 'June 2025'): 
Learning styles: visual / auditory / reading / kinesthetic / mixed
Enter your learning style: 
```

---

## 📥 Sample Input

```
Enter subject: Machine Learning
Enter your goal: Become job-ready as a junior ML engineer
Enter available study hours per day: 2
Enter deadline: 30 days
Enter your learning style: visual
```

---

## 📤 Sample Output

```
============================================================
[Agent 1] Analyzing requirements...
[Agent 1] ✓ Requirements analyzed successfully.

[Agent 2] Researching topics and curriculum...
[Agent 2] ✓ Topic research completed successfully.

[Agent 3] Creating personalized day-wise study plan...
[Agent 3] ✓ Study plan created successfully.

[Agent 4] Reviewing and finalizing the study plan...
[Agent 4] ✓ Review and optimization completed.

╔══════════════════════════════════════════════════════════════╗
║           PERSONALIZED STUDY PLAN - FINAL VERSION           ║
╚══════════════════════════════════════════════════════════════╝

📋 REQUIREMENTS SUMMARY
...

🎯 LEARNING OBJECTIVES
...

📚 TOPIC ROADMAP
Week 1: Python & Math Foundations
Week 2: Core ML Algorithms
Week 3: Practical Implementation
Week 4: Projects & Interview Prep

📅 DAY-WISE STUDY SCHEDULE
Day 1: Python review + NumPy basics (2 hrs)
Day 2: Pandas + data manipulation (2 hrs)
...

🔄 REVISION STRATEGY
...

🛠️ PRACTICE PROJECTS
...

💡 STUDY TIPS & STRATEGIES
...

✅ Your personalized study plan is ready!
```

---

## 🧪 Demo Instructions

1. Ensure your `.env` file has a valid `GOOGLE_API_KEY`
2. Run `python multi_agent_system.py`
3. Enter the sample inputs above
4. Watch each agent process in real-time
5. Copy the final output to a document for your study reference

---

## 📦 Project Structure

```
ai-study-planner/
│
├── multi_agent_system.py   # Main application (all agents + graph)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .env.example            # Environment variable template
├── .gitignore              # Git ignore rules
└── DEMO_SCRIPT.md          # Video demo voice-over script
```

---

## 🎓 Learning Outcomes

By studying this project, you will understand:

- **LangGraph StateGraph** — How to build multi-step agent pipelines with shared state
- **LangChain Google Gemini** — How to use Gemini models through LangChain's abstraction
- **TypedDict State** — How to define and pass typed shared state between nodes
- **Agent Design Patterns** — How to decompose a complex task into specialized agents
- **Prompt Engineering** — How to write effective system and human prompts for each role
- **Error Propagation** — How to handle failures gracefully in a multi-step pipeline
- **Production Code Practices** — Type hints, docstrings, input validation, clean structure

---

## ⚙️ Configuration

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Your Google Gemini API key | ✅ Yes |

**Get your FREE Gemini API key:**
1. Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy and paste it into your `.env` file

The system uses `gemini-1.5-flash` by default (fast + free tier friendly). To use a more powerful model, change it in `get_llm()`:
```python
model="gemini-1.5-pro"
```

---

## 🔒 Security Notes

- Never commit your `.env` file (it's in `.gitignore`)
- Never hardcode API keys in source code
- The `.env.example` file is safe to commit — it contains no real secrets

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙋 Support

If you encounter issues:
1. Verify your `GOOGLE_API_KEY` is valid and active
2. Ensure all dependencies are installed: `pip install -r requirements.txt`
3. Check Python version: `python --version` (must be 3.9+)
4. Verify your Gemini API key is valid at [Google AI Studio](https://aistudio.google.com/app/apikey)
