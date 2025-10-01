from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    task = "text-generation",

)

model = ChatHuggingFace(llm=llm)




loader = PyPDFLoader("Project.pdf")
documents = loader.load() 

resume_text = "\n".join([doc.page_content for doc in documents])



sentence = """Job Title: Software Engineer – Backend

Job Description:
We are seeking a highly skilled Software Engineer to join our backend development team. The ideal candidate will have experience in building scalable web applications, RESTful APIs, and microservices architecture.

Responsibilities:


Design, develop, and maintain backend services using Java, Python, or Node.js.

Implement and maintain database systems (SQL and NoSQL).

Collaborate with frontend developers to integrate user-facing elements.

Optimize application performance and ensure security best practices.

Write unit tests and participate in code reviews.

Requirements:

Bachelor’s degree in Computer Science, Software Engineering, or a related field.

3+ years of experience in backend development.

Strong knowledge of RESTful APIs, microservices, and cloud platforms (AWS, Azure, or GCP).

Experience with Docker, Kubernetes, and CI/CD pipelines.

Familiarity with Agile methodologies.

Excellent problem-solving and communication skills.

Preferred Qualifications:

Experience with machine learning, data pipelines, or big data technologies.

Knowledge of message queues (RabbitMQ, Kafka) and event-driven architecture."""



template = """You are an expert in resume optimization and ATS (Applicant Tracking System) keyword extraction.

I will provide you a job description. Your task is to analyze the text and extract only the most important keywords and phrases that are crucial for ATS matching. These may include skills, technologies, tools, certifications, programming languages, job-specific terms, action verbs, and qualifications.

Instructions:

Only return the extracted keywords and phrases as a comma-separated list.

Do not include any extra explanation, summary, or filler text.

Focus on words that are highly relevant for ATS parsing, not generic words.

Preserve multi-word terms exactly as they appear in the job description (e.g., “machine learning”, “project management”).

Avoid duplicates.

Job Description:
 
""" + sentence 

keywords = model.invoke(template)


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

prompt1 = """convert the below latex code to latex code as per overleaf with pdfLaTeX extra text or explanation. + {result.content}"""


final_result = model.invoke(prompt1)
