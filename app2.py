import streamlit as st
from fpdf import FPDF
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
import re
import plotly.graph_objects as go
import tempfile
import os
from openai import OpenAI

# ---------------------------------------------------
# Setup AI API (OpenRouter Integration)
# ---------------------------------------------------
OPENAI_API_KEY = "your api key "
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENAI_API_KEY
) if OPENAI_API_KEY else None

# ---------------------------------------------------
# Load SBERT Model (STRICTLY UNCHANGED)
# ---------------------------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
model = load_model()

# ---------------------------------------------------
# Extended Job Role Knowledge Base (UNCHANGED)
# ---------------------------------------------------
JOB_ROLES = {
    "frontend developer": "Frontend Developer skilled in React, Vue, Angular, JavaScript, HTML, CSS, Responsive Design, UI Components, REST APIs.",
    "backend developer": "Backend Developer skilled in Node.js, Python, Go, Java, APIs, Databases, Microservices, Authentication, System Architecture.",
    "full stack developer": "Full Stack Developer skilled in React, Node.js, Express, MongoDB, SQL, REST APIs, Deployment, Git.",
    "mobile app developer": "Mobile Developer skilled in iOS, Swift, Android, Kotlin, App Deployment, Mobile UI, API Integration.",
    "cross platform developer": "Cross Platform Developer skilled in Flutter, React Native, Mobile Architecture, API Integration.",
    "embedded systems engineer": "Embedded Engineer skilled in C, C++, Firmware, Microcontrollers, IoT Systems.",
    "game developer": "Game Developer skilled in Unity, Unreal Engine, C#, Game Physics, Rendering.",
    "desktop application developer": "Desktop Developer skilled in .NET, Electron, C#, Windows Applications.",
    "api developer": "API Developer skilled in REST, GraphQL, gRPC, Backend Services.",
    "graphics engineer": "Graphics Engineer skilled in OpenGL, Vulkan, Shaders, Rendering Pipelines.",
    "data scientist": "Data Scientist skilled in Python, R, Statistical Modeling, Machine Learning, SQL, Pandas, NumPy, Data Visualization.",
    "machine learning engineer": "Machine Learning Engineer skilled in PyTorch, TensorFlow, Model Deployment, Feature Engineering.",
    "data engineer": "Data Engineer skilled in ETL Pipelines, Spark, Snowflake, Big Data Systems.",
    "ai research scientist": "AI Research Scientist skilled in Deep Learning, Neural Networks, Research Publications.",
    "nlp engineer": "NLP Engineer skilled in Language Models, Transformers, Text Processing.",
    "computer vision engineer": "Computer Vision Engineer skilled in OpenCV, Image Processing, Deep Learning.",
    "data analyst": "Data Analyst skilled in SQL, Tableau, PowerBI, Reporting.",
    "business intelligence developer": "BI Developer skilled in Data Warehousing, Dashboards, SQL Analytics.",
    "ai prompt engineer": "Prompt Engineer skilled in LLM Optimization, Generative AI Systems.",
    "mlops engineer": "MLOps Engineer skilled in ML Lifecycle, CI/CD for ML, Model Monitoring.",
    "devops engineer": "DevOps Engineer skilled in CI/CD, Docker, Jenkins, Kubernetes.",
    "cloud architect": "Cloud Architect skilled in AWS, Azure, GCP, Infrastructure Design.",
    "site reliability engineer": "SRE skilled in Monitoring, Scalability, System Uptime.",
    "platform engineer": "Platform Engineer skilled in Internal Developer Platforms.",
    "systems administrator": "Systems Administrator skilled in Linux, Windows Server Management.",
    "network engineer": "Network Engineer skilled in Routing, Switching, Firewalls.",
    "database administrator": "DBA skilled in Postgres, Oracle, NoSQL Databases.",
    "security engineer": "Security Engineer skilled in Penetration Testing, Encryption.",
    "cloud security specialist": "Cloud Security Specialist skilled in Cloud Compliance.",
    "iac specialist": "Infrastructure as Code Specialist skilled in Terraform, Automation.",
    "ui ux designer": "UI UX Designer skilled in Figma, Adobe XD, User Research.",
    "product manager": "Technical Product Manager skilled in Roadmaps, Agile, Feature Planning.",
    "qa automation engineer": "QA Automation Engineer skilled in Selenium, Cypress.",
    "manual test engineer": "Manual Test Engineer skilled in User Acceptance Testing.",
    "sdet": "SDET skilled in Automation Framework Development.",
    "systems analyst": "Systems Analyst bridging Business and Technology.",
    "solutions architect": "Solutions Architect skilled in Client-Facing Technical Design.",
    "user researcher": "User Researcher skilled in Behavior Analysis.",
    "technical writer": "Technical Writer skilled in API Documentation.",
    "accessibility specialist": "Accessibility Specialist skilled in ADA WCAG Compliance.",
    "blockchain developer": "Blockchain Developer skilled in Smart Contracts, Solidity.",
    "cybersecurity analyst": "Cybersecurity Analyst skilled in SOC, Threat Detection.",
    "erp consultant": "ERP Consultant skilled in SAP, Oracle NetSuite.",
    "sales engineer": "Sales Engineer skilled in Technical Pre-Sales.",
    "scrum master": "Scrum Master skilled in Agile Methodology.",
    "engineering manager": "Engineering Manager skilled in Team Leadership.",
    "cto": "Chief Technology Officer leading Technology Strategy.",
    "it project manager": "IT Project Manager skilled in Delivery Management.",
    "digital forensic examiner": "Digital Forensic Examiner skilled in Cybercrime Investigation.",
    "ar vr developer": "AR VR Developer skilled in Metaverse, Immersive Technologies, 3D Interaction."
}

# ---------------------------------------------------
# NEW: Resume Completion Progress Helper (pure addition - no logic changed)
# ---------------------------------------------------
def calculate_resume_progress(name, summary, skills, experience, education, projects, achievements, certifications, other_links):
    score = 20  # base for personal details
    if summary and len(summary.strip()) > 40:
        score += 20
    if skills and len(skills.strip()) > 10:
        score += 15
    if experience and len(experience.strip()) > 50:
        score += 25
    if education and len(education.strip()) > 20:
        score += 10
    if projects or achievements or certifications or other_links:
        score += 10
    return min(100, score)

# ---------------------------------------------------
# Circular Gauge Function (UNCHANGED)
# ---------------------------------------------------
def circular_score(title, score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': title, 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#1E88E5"},
            'steps': [
                {'range': [0, 50], 'color': "#FFCDD2"},
                {'range': [50, 75], 'color': "#FFF9C4"},
                {'range': [75, 100], 'color': "#C8E6C9"}
            ],
        }
    ))
    fig.update_layout(
        height=250, 
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# ---------------------------------------------------
# Helper Functions (UNCHANGED)
# ---------------------------------------------------
def extract_text(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return " ".join(text.split())

def keyword_match_percentage(jd_text, resume_text):
    jd_words = set(clean_text(jd_text).split())
    resume_words = set(clean_text(resume_text).split())
    matched = jd_words.intersection(resume_words)
    if len(jd_words) == 0:
        return 0
    return round((len(matched) / len(jd_words)) * 100, 1)

# ---------------------------------------------------
# AI Suggestions Helper (UNCHANGED)
# ---------------------------------------------------
def get_ai_suggestions(resume_text, target_role, jd_text):
    if not OPENAI_API_KEY or not client:
        return "API Key not configured. Please set the OPENAI_API_KEY environment variable."
    
    try:
        prompt = f"""
        You are an expert ATS and career coach. I am providing you with a resume text and the required skills for a target role.
        
        Target Role: {target_role}
        Required Skills / Job Description: {jd_text}
        
        Resume Text:
        {resume_text}
        
        Please analyze the resume against the job description and provide structured feedback using the following markdown format. Do NOT score the resume, only provide suggestions:
        
        ### Weak Areas
        (Identify gaps in experience, formatting issues, or lacking context)
        
        ### Missing Keywords
        (List specific ATS keywords from the job description that are missing in the resume)
        
        ### Improvement Suggestions
        (Provide actionable tips to make the resume stronger)
        
        ### Improved Professional Summary
        (Write a strong, ATS-optimized 3-4 sentence professional summary tailored to this role based on their current experience)
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional career coach and ATS optimization assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred while communicating with OpenAI API: {e}"

# ---------------------------------------------------
# PDF Generation Function (FIXED X-CURSOR - UNCHANGED)
# ---------------------------------------------------
def generate_pdf(name, email, phone, summary, experience, education, skills, template="Professional"):
    pdf = FPDF(format='A4')
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()
    pdf.set_margins(10, 10, 10)
    
    MAX_Y = 275
    
    def add_safe_section(title, body, font_family, header_r=0, header_g=0, header_b=0, is_fill=False):
        if not body.strip() or pdf.get_y() >= MAX_Y:
            return
        
        pdf.set_x(10)
        
        pdf.set_font(font_family, "B", 11)
        pdf.set_text_color(header_r, header_g, header_b)
        if is_fill:
            pdf.set_fill_color(230, 230, 230)
            pdf.cell(0, 6, f" {title.upper()}", ln=True, align="L", fill=True)
        else:
            pdf.cell(0, 6, title.upper(), ln=True, align="L")
            pdf.set_draw_color(header_r, header_g, header_b)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(1)
        
        pdf.set_font(font_family, "", 9)
        pdf.set_text_color(30, 30, 30)
        available_width = 190
        
        for orig_line in body.split('\n'):
            if not orig_line.strip():
                continue
            
            line = orig_line.strip()
            if line.startswith('-'):
                line = f"- {line[1:].strip()}"
            
            if pdf.get_y() >= MAX_Y - 5:
                pdf.set_x(10)
                pdf.set_font(font_family, "I", 8)
                pdf.cell(0, 4, "[Content truncated to fit single page]", ln=True)
                break
            
            text_width = pdf.get_string_width(line)
            num_lines_est = max(1, int((text_width - 1) // available_width) + 1)
            est_height = num_lines_est * 4.5 + 1
            
            if pdf.get_y() + est_height > MAX_Y:
                pdf.set_x(10)
                pdf.set_font(font_family, "I", 8)
                pdf.cell(0, 4, "[Content truncated to fit single page]", ln=True)
                break
            
            pdf.set_x(10)
            pdf.multi_cell(available_width, 4.5, line)
        
        pdf.ln(2)

    pdf.set_x(10)
    if template == "Modern":
        pdf.set_font("Arial", "B", 18)
        pdf.set_text_color(0, 51, 102)
        pdf.cell(0, 8, name.upper(), ln=True, align="L")
        pdf.set_font("Arial", "", 10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 5, f"{email} | {phone}", ln=True, align="L")
        pdf.ln(2)
        add_section = lambda t, b: add_safe_section(t, b, "Arial", 0, 51, 102)
    elif template == "Professional":
        pdf.set_font("Helvetica", "B", 20)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 8, name, ln=True, align="C")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(70, 70, 70)
        pdf.cell(0, 5, f"{email} | {phone}", ln=True, align="C")
        pdf.ln(2)
        add_section = lambda t, b: add_safe_section(t, b, "Helvetica", 40, 40, 40)
    elif template == "Executive":
        pdf.set_font("Times", "B", 20)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 8, name.upper(), ln=True, align="C")
        pdf.set_font("Times", "I", 10)
        pdf.cell(0, 5, f"{email} | {phone}", ln=True, align="C")
        pdf.set_draw_color(0, 0, 0)
        pdf.set_line_width(0.5)
        pdf.line(10, pdf.get_y()+2, 200, pdf.get_y()+2)
        pdf.set_line_width(0.2)
        pdf.ln(3)
        add_section = lambda t, b: add_safe_section(t, b, "Times", 0, 0, 0)
    elif template == "Creative":
        pdf.set_font("Arial", "B", 22)
        pdf.set_text_color(41, 128, 185)
        pdf.cell(0, 8, name, ln=True, align="R")
        pdf.set_font("Arial", "", 10)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 5, f"{email} | {phone}", ln=True, align="R")
        pdf.ln(3)
        add_section = lambda t, b: add_safe_section(t, b, "Arial", 41, 128, 185)
    elif template == "Compact":
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 6, name.upper(), ln=True, align="L")
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 4, f"{email} | {phone}", ln=True, align="L")
        pdf.ln(2)
        add_section = lambda t, b: add_safe_section(t, b, "Helvetica", 0, 0, 0, is_fill=True)
    else:
        pdf.set_font("Times", "B", 18)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 8, name, ln=True, align="C")
        pdf.set_font("Times", "", 10)
        pdf.cell(0, 5, f"{email} | {phone}", ln=True, align="C")
        pdf.ln(2)
        add_section = lambda t, b: add_safe_section(t, b, "Times", 0, 0, 0)
    
    add_section("Professional Summary", summary)
    add_section("Skills", skills)
    add_section("Experience", experience)
    add_section("Education", education)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        with open(tmp.name, "rb") as f:
            pdf_bytes = f.read()
    os.unlink(tmp.name)
    return pdf_bytes

# ---------------------------------------------------
# Streamlit UI Configuration
# ---------------------------------------------------
st.set_page_config(page_title="AI Resume Builder & ATS", layout="wide", page_icon="📄", initial_sidebar_state="expanded")

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# Dynamic CSS Injection for Light / Dark Mode
st.markdown(f"""
    <style>
    {
    '''
    /* 🌙 DARK MODE CSS */
    .stApp, [data-testid="stAppViewContainer"] {
        background: #0A1428 !important;
    }
    [data-testid="stSidebar"] {
        background: #0A1428 !important;
        border-right: 1px solid #1E3A8A !important;
    }
    [data-testid="stHeader"] {
        background: rgba(10, 20, 40, 0.8) !important;
    }
    /* Cards, Expanders, and Bordered Containers */
    [data-testid="stExpander"] div[role="button"],
    [data-testid="stExpander"] div[data-testid="stExpanderDetails"],
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #112233 !important;
        border: 1px solid #1E3A8A !important;
        border-radius: 8px !important;
        color: #E2E8F0 !important;
    }
    /* Global Text Elements */
    .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp label, .stApp span, .stApp div.stMarkdown, .stApp .stText {
        color: #E2E8F0 !important;
    }
    /* Protect Alerts */
    [data-testid="stAlert"] * {
        color: inherit !important;
    }
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #64B5F6 !important;
    }
    /* Secondary Buttons */
    .stButton > button {
        background-color: #112233 !important;
        border: 1px solid #64B5F6 !important;
        color: #64B5F6 !important;
    }
    .stButton > button:hover {
        background-color: #64B5F6 !important;
        color: #0A1428 !important;
    }
    /* Primary Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #1E3A8A, #64B5F6) !important;
        color: white !important;
        border: none !important;
    }
    /* Inputs */
    input, textarea, div[data-baseweb="select"] > div, div[data-baseweb="popover"] > div {
        background-color: #0A1428 !important;
        color: #E2E8F0 !important;
        border: 1px solid #1E3A8A !important;
    }
    ''' if st.session_state.dark_mode else '''
    /* ☀️ LIGHT MODE CSS */
    .stApp, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #E3F2FD 0%, #F8FAFC 100%) !important;
    }
    [data-testid="stSidebar"] {
        background: rgba(243, 248, 255, 0.8) !important;
        border-right: 1px solid #BBDEFB !important;
    }
    [data-testid="stHeader"] {
        background: rgba(227, 242, 253, 0.8) !important;
    }
    /* Cards, Expanders, and Bordered Containers */
    [data-testid="stExpander"] div[role="button"],
    [data-testid="stExpander"] div[data-testid="stExpanderDetails"],
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: rgba(255, 255, 255, 0.85) !important;
        border: 1px solid #BBDEFB !important;
        border-radius: 8px !important;
    }
    /* Primary Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #1E88E5, #64B5F6) !important;
        color: white !important;
        border: none !important;
    }
    '''
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.toggle("🌙 Dark Mode", key="dark_mode")
    st.divider()
    
    st.title("⚙️ AI Configuration")
    st.divider()
    template = st.selectbox("🎨 Choose Template", ["Professional", "Executive", "Creative", "Modern", "Minimalist", "Compact"], help="Choose a professional single-page design")
    target_role = st.selectbox("🎯 Target Job Role", list(JOB_ROLES.keys()), index=2, help="This powers all ATS scoring & keyword insights")
    st.divider()
    st.info("💡 Pro Tip: The more complete your resume, the higher your ATS score will be.")

# Main Header
st.markdown("""
    <h1 style="text-align:center; background: linear-gradient(90deg, #0A2540, #1E88E5); -webkit-background-clip:text; -webkit-text-fill-color:transparent; font-size:3rem; margin-bottom:0;">
        📄 AI Resume Builder
    </h1>
    <h2 style="text-align:center; color:#1E88E5; margin-top:0;">Build • Score • Optimize with Real-Time ATS Intelligence</h2>
""", unsafe_allow_html=True)
st.divider()

# Tabs
tab1, tab2 = st.tabs(["🏗️ Build & Enhance Resume", "📂 Upload & Analyze External PDF"])

with tab1:
    st.info("✨ Fill the sections below. Your **Live Resume Preview** and **Insights Panel** update instantly as you type.")

    # MAIN LAYOUT: LEFT (Builder) + RIGHT (Live Preview + Insights)
    left_col, right_col = st.columns([3, 2])

    # LEFT COLUMN - All sections now in clean expanders
    with left_col:
        st.subheader("📝 Resume Builder")

        with st.expander("👤 Personal Details", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                name = st.text_input("Full Name", "John Doe", key="name_input")
            with c2:
                phone = st.text_input("Phone", "+1 234 567 890", key="phone_input")
            email = st.text_input("Email", "john@example.com", key="email_input")
            st.caption("💼 Make sure your contact details look professional.")

        with st.expander("💼 Professional Summary", expanded=True):
            summary = st.text_area("Professional Summary", 
                                  "Results-driven professional with experience in software development. Proven track record of delivering high-quality web applications using modern technologies.",
                                  height=110, key="summary_input")
            st.caption("Keep 3-4 powerful sentences focused on impact.")

        with st.expander("🛠️ Skills", expanded=True):
            popular_skills = ["React", "Node.js", "Python", "JavaScript", "SQL", "Docker", "Kubernetes", "AWS", "Machine Learning", 
                            "TensorFlow", "PyTorch", "Git", "HTML", "CSS", "Flutter", "Swift", "Kotlin", "MongoDB", "Express", "Redis"]
            selected_skills = st.multiselect("Smart Skill Suggestions", popular_skills, 
                                           default=["React", "Node.js", "Python"], 
                                           help="Select from popular skills or type your own below")
            custom_skills = st.text_input("Additional skills (comma separated)", "", key="custom_skills_input")
            skills = ", ".join(selected_skills)
            if custom_skills.strip():
                skills += ", " + custom_skills
            st.caption("💡 These skills feed directly into ATS scoring and skill-gap analysis.")

        with st.expander("🏢 Experience", expanded=True):
            experience = st.text_area("Work Experience", 
                                     "Senior Developer at Tech Corp (2020 - Present)\n- Led a team of 5 engineers to deliver a new SaaS product.\n- Optimized backend performance by 40% using Redis.\n\nSoftware Engineer at DevStudio (2018 - 2020)\n- Developed interactive frontend components using React.",
                                     height=160, key="experience_input")
            st.caption("Use bullet points with metrics for maximum impact.")

        with st.expander("🎓 Education", expanded=True):
            education = st.text_area("Education", 
                                    "B.S. in Computer Science\nState University - Graduated 2018\nGPA: 3.8/4.0",
                                    height=100, key="education_input")

        with st.expander("💻 Projects (Optional)"):
            projects = st.text_area("Projects", 
                                   "E-commerce Platform: Built a scalable microservices architecture.\nAI Chatbot: Integrated OpenAI API for customer support.",
                                   height=100, key="projects_input")

        with st.expander("🏆 Achievements (Optional)"):
            achievements = st.text_area("Achievements", "Employee of the Year 2022\nReduced cloud costs by 30%", height=80, key="achievements_input")

        with st.expander("📜 Certifications (Optional)"):
            certifications = st.text_area("Certifications", "AWS Certified Solutions Architect\nCertified Kubernetes Administrator", height=80, key="certifications_input")

        with st.expander("🔗 Other Links (Optional)"):
            other_links = st.text_area("Links (LinkedIn, GitHub, Portfolio)", 
                                      "LinkedIn: linkedin.com/in/johndoe\nGitHub: github.com/johndoe", 
                                      height=80, key="links_input")

    # RIGHT COLUMN - Live Preview + SaaS Insights Panel
    with right_col:
        # Live Resume Preview
        st.subheader("📄 Live Resume Preview")
        with st.container(border=True, height=620):
            st.markdown(f"**{name.upper()}**")
            st.markdown(f"{email} | {phone}")
            st.divider()
            st.markdown("**💼 Professional Summary**")
            st.write(summary if summary else "Your summary will appear here...")
            st.divider()
            st.markdown("**🛠️ Skills**")
            st.write(skills if skills else "Your skills will appear here...")
            st.divider()
            st.markdown("**🏢 Experience**")
            st.write(experience.replace("\n", "\n\n") if experience else "Experience section will appear here...")
            st.divider()
            st.markdown("**🎓 Education**")
            st.write(education if education else "Education will appear here...")
            if projects:
                st.divider()
                st.markdown("**💻 Projects**")
                st.write(projects)

        # Resume Completion Progress
        st.subheader("📈 Resume Completion")
        progress_value = calculate_resume_progress(name, summary, skills, experience, education, projects, achievements, certifications, other_links)
        st.progress(progress_value / 100)
        st.caption(f"**{progress_value}% Complete** — Almost ready for ATS submission!")

        # Role Insights
        st.subheader("🎯 Role Insights")
        st.info(f"**{target_role.title()}**\n\n{JOB_ROLES[target_role]}")

        # Top ATS Keywords
        st.subheader("🔑 Top ATS Keywords")
        role_keywords = [kw.strip() for kw in JOB_ROLES[target_role].split(",") if kw.strip()]
        for kw in role_keywords[:12]:
            st.markdown(f"• {kw}")

        # Resume Strength Metrics
        st.subheader("💪 Resume Strength Metrics")
        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            skill_count = len([s for s in skills.split(",") if s.strip()]) if skills else 0
            st.metric("Skills", skill_count)
        with mcol2:
            exp_count = max(1, len([line for line in experience.split("\n") if line.strip() and not line.startswith("-")]))
            st.metric("Experience Entries", exp_count)
        with mcol3:
            st.metric("Projects / Achievements", 1 if projects or achievements else 0)

        # Skill Gap Analyzer
        st.subheader("🔍 Skill Gap Analyzer")
        role_skills_set = set(word.strip().lower() for word in JOB_ROLES[target_role].split(","))
        user_skills_set = set(s.strip().lower() for s in skills.split(",") if s.strip())
        missing_skills = list(role_skills_set - user_skills_set)[:8]
        if missing_skills:
            st.warning("**Missing for this role:**\n" + "\n".join(f"• {s.title()}" for s in missing_skills))
        else:
            st.success("✅ Excellent skill alignment with target role!")

    # CENTERED ACTION BUTTON (below the two-column layout)
    st.divider()
    left_space, center_button, right_space = st.columns([1, 2, 1])
    with center_button:
        run_ai_suggestions = st.checkbox("🧠 Include Actionable AI Suggestions", value=True)
        generate_btn = st.button("🚀 Generate Resume & Run Full ATS Analysis", type="primary", use_container_width=True)

    # RESULTS SECTION (appears only after generation)
    if generate_btn:
        with st.spinner("⏳ Generating PDF • Running semantic analysis • Calculating ATS score..."):
            pdf_data = generate_pdf(name, email, phone, summary, experience, education, skills, template)
            
            st.success("🎉 One-Page Resume Generated Successfully!")
            dl_col1, dl_col2, dl_col3 = st.columns([1, 2, 1])
            with dl_col2:
                st.download_button(
                    label="⬇️ Download Your Tailored PDF Resume",
                    data=pdf_data,
                    file_name=f"{name.replace(' ', '_')}_Resume.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

            st.divider()
            st.markdown(f"### 📊 ATS Evaluation Results (Target: **{target_role.title()}**)")

            # Resume Strength Dashboard (new)
            st.subheader("💪 Resume Strength Dashboard")
            strength_col1, strength_col2, strength_col3, strength_col4 = st.columns(4)
            with strength_col1:
                st.metric("Skills Count", len([s for s in skills.split(",") if s.strip()]))
            with strength_col2:
                st.metric("Experience Entries", max(1, len(experience.splitlines()) // 2))
            with strength_col3:
                st.metric("Projects / Achievements", bool(projects or achievements))
            with strength_col4:
                st.metric("Resume Length", "Optimal" if len(summary) + len(experience) < 1800 else "Needs shortening")

            # ATS Gauges
            generated_resume_text = f"{name} {email} {phone} {summary} {skills} {experience} {projects} {achievements} {certifications} {education} {other_links}"
            jd_text = JOB_ROLES[target_role]
            
            emb = model.encode([clean_text(jd_text), clean_text(generated_resume_text)])
            semantic_score = round(float(cosine_similarity([emb[0]], [emb[1]])[0][0]) * 100, 1)
            keyword_score = keyword_match_percentage(jd_text, generated_resume_text)
            
            g1, g2 = st.columns(2)
            with g1:
                st.plotly_chart(circular_score("Semantic Match", semantic_score), use_container_width=True)
                label = "Excellent" if semantic_score > 75 else "Good" if semantic_score > 50 else "Needs Improvement"
                st.success(f"**{label}**") if semantic_score > 75 else st.warning(f"**{label}**") if semantic_score > 50 else st.error(f"**{label}**")
            with g2:
                st.plotly_chart(circular_score("Keyword Match", keyword_score), use_container_width=True)
                label = "Excellent" if keyword_score > 75 else "Good" if keyword_score > 50 else "Needs Improvement"
                st.success(f"**{label}**") if keyword_score > 75 else st.warning(f"**{label}**") if keyword_score > 50 else st.error(f"**{label}**")

            # Matched vs Missing Keywords
            jd_set = set(clean_text(jd_text).split())
            resume_set = set(clean_text(generated_resume_text).split())
            matched_kw = sorted(list(jd_set.intersection(resume_set)))[:12]
            missing_kw = sorted(list(jd_set - resume_set))[:12]
            
            st.subheader("✅ Matched & Missing Keywords")
            mk1, mk2 = st.columns(2)
            with mk1:
                st.success("**Matched Keywords**")
                st.write(", ".join(matched_kw) if matched_kw else "None yet")
            with mk2:
                st.warning("**Missing Keywords**")
                st.write(", ".join(missing_kw) if missing_kw else "None — Perfect match!")

            # AI Suggestions
            if run_ai_suggestions:
                st.divider()
                st.markdown("### 🤖 OpenAI Actionable Feedback & Improvement Tips")
                if not OPENAI_API_KEY:
                    st.warning("⚠️ API Key not configured.")
                else:
                    ai_feedback = get_ai_suggestions(generated_resume_text, target_role, jd_text)
                    with st.container(border=True):
                        st.markdown(ai_feedback)

with tab2:
    st.info("📂 Have an existing resume? Upload it for instant ATS scoring against your target role.")
    st.markdown("### 📤 Upload Existing Resume for Analysis")
    uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf"])
    
    left_up, center_up, right_up = st.columns([1, 2, 1])
    with center_up:
        run_ai_suggestions_upload = st.checkbox("🧠 Include AI Suggestions Analysis (Uploaded)", value=True)
        analyze_btn = st.button("📊 Analyze Uploaded Resume", type="primary", use_container_width=True)
        
    if analyze_btn:
        if uploaded_file:
            with st.spinner("⏳ Extracting text and running full ATS analysis..."):
                raw_resume = extract_text(uploaded_file)
                jd_text = JOB_ROLES[target_role]
                emb = model.encode([clean_text(jd_text), clean_text(raw_resume)])
                semantic_score = round(float(cosine_similarity([emb[0]], [emb[1]])[0][0]) * 100, 1)
                keyword_score = keyword_match_percentage(jd_text, raw_resume)
                
                st.divider()
                st.markdown(f"### 📊 Analysis Results for **{target_role.title()}**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(circular_score("Semantic ATS Score", semantic_score), use_container_width=True)
                with col2:
                    st.plotly_chart(circular_score("Keyword Match Score", keyword_score), use_container_width=True)
                
                if semantic_score > 75:
                    st.success("🌟 **Strong semantic alignment detected!**")
                elif semantic_score > 50:
                    st.warning("⚠️ **Moderate semantic alignment.**")
                else:
                    st.error("❗ **Low semantic alignment.**")
                
                if run_ai_suggestions_upload:
                    st.divider()
                    st.markdown("### 🤖 AI Resume Enhancements")
                    if not OPENAI_API_KEY:
                        st.warning("⚠️ API Key not found.")
                    else:
                        ai_feedback = get_ai_suggestions(raw_resume, target_role, jd_text)
                        with st.container(border=True):
                            st.markdown(ai_feedback)
        else:
            st.warning("⚠️ Please upload a PDF file to analyze.")
