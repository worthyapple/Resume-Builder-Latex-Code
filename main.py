import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
import os
import tempfile

# Load environment variables
load_dotenv()

# Initialize LLM
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-480B-A35B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)

st.title("ATS Resume Optimizer")

# Upload PDF
uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf"])

# Paste job description
job_description = st.text_area("Paste Job Description Here", height=250)

if uploaded_file and job_description:
    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # Load PDF using original logic
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    resume_text = "\n".join([doc.page_content for doc in documents])

    st.success("PDF loaded successfully!")

    # Original keyword extraction template
    template = f"""You are an expert in resume optimization and ATS (Applicant Tracking System) keyword extraction.

I will provide you a job description. Your task is to analyze the text and extract only the most important keywords and phrases that are crucial for ATS matching. These may include skills, technologies, tools, certifications, programming languages, job-specific terms, action verbs, and qualifications.

Instructions:

Only return the extracted keywords and phrases as a comma-separated list.

Do not include any extra explanation, summary, or filler text.

Focus on words that are highly relevant for ATS parsing, not generic words.

Preserve multi-word terms exactly as they appear in the job description (e.g., “machine learning”, “project management”).

Avoid duplicates.

Job Description:
{job_description}"""

    keywords = model.invoke(template)
    st.subheader("Extracted ATS Keywords")
    st.text(keywords)

    # Original resume rewrite prompt
    prompt = f"""You are an expert in resume writing and ATS optimization. I will provide:

A candidate’s existing resume or details (education, experience, projects, skills).

A list of ATS keywords.

Your task is to rewrite the resume to meet the following requirements:

Requirements:

The resume must be strictly one page. highlight the most relevant information.

if links are there ensure they are clickable in the PDF.

Preserve all the sections in the original resume (e.g., Contact, Summary, Experience, Education, Skills, Projects, Certifications).

Format it in clean, ATS-friendly style, suitable for PDF export: simple text, no fancy tables, graphics, or columns.

Incorporate all provided ATS keywords naturally into the resume wherever relevant.

Use strong action verbs, quantifiable achievements, and relevant metrics in experience/project descriptions.

Avoid filler words or irrelevant details.

Ensure each section is clearly labeled and organized in standard resume order.

Nothing should be left for suggestion take information from the existing resume.

Output:

Provide the resume as latex code as per Overleaf with pdfLaTeX(strictly), ready to be converted to PDF.

Do not add extra explanations, instructions, or notes.

Candidate Details / Existing Resume:
{resume_text}

ATS Keywords:
{keywords}"""

    result = model.invoke(prompt)

    # Original LaTeX formatting step
    prompt1 = f"""convert the below latex code to latex code as per Overleaf with pdfLaTeX, no extra text or explanation. + {result.content}"""
    final_result = model.invoke(prompt1)

    st.subheader("ATS-Optimized LaTeX Code")

    # Display code block for direct copy
    st.code(final_result, language="latex")

