from mcp.server.fastmcp import FastMCP
import requests as rq
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from os import environ as ENV
from typing import List
from llama_index.core.llms import ChatMessage
import json
from llama_index.llms.groq import Groq
import argparse

load_dotenv()

class JobMatchEvaluation(BaseModel):
    match_score: int = Field(description="An evaluation, between 0 and 100, of how much the job details match the resume data from the candidate")
    reasons: str = Field(description="Reasons for the evaluation")

mcp = FastMCP(name = "Resume Matcher MCP")
llm = Groq(model="qwen-qwq-32b", api_key=ENV["groq_api_key"])
llm_struct = llm.as_structured_llm(JobMatchEvaluation)



@mcp.tool(name="job_searcher", description="Search for a job with a given title and location. Requires: job_title (List[str]) - the job titles you are searching for, it's a list with one or more elements.")
def job_searcher(job_title: List[str]):
    response = rq.post(
        "https://api.theirstack.com/v1/jobs/search",
        headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ENV['theirstack_api_key']}"
        },
        json={
        "page": 0,
        "limit": 5,
        "job_title_or": job_title,
        "posted_at_max_age_days": 1,
        "company_country_code_or": [
            "AT",
            "BE",
            "BG",
            "CY",
            "CZ",
            "DE",
            "DK",
            "EE",
            "ES",
            "FI",
            "FR",
            "GR",
            "HR",
            "HU",
            "IE",
            "IT",
            "LT",
            "LU",
            "LV",
            "MT",
            "NL",
            "PL",
            "PT",
            "RO",
            "SE",
            "SI",
            "SK"
        ]
        }
    )
    json_response = response.json()
    data = json_response["data"]
    return json.dumps(data)

@mcp.tool(name="evaluate_job_match", description = "Evaluates the match between jobs opening and the candidate's profile. Requires as input: candidate_profile (str) - the data acquired from the candidate's resume with the 'resume_parser' tool, jobs (str): a JSON-like string of the job openings, retrieved from a job search with the 'job_searcher' tool")
async def evaluate_job_match(candidate_profile: str, jobs: str):
    jobs_list = json.loads(jobs)
    base_messages = [ChatMessage.from_str(role="system", content="You are a job matching assistant. Your task is to evaluate a job based on its match with the candidate's profile, taking into account the job title, the skills required, the seniority level, the physical location (where the company offering the work is based in) and the working location (remote/hybrid/on-site). You then have to produce a match score (between 0 and 100) and justify that match scores explaining your reasons for that."), ChatMessage.from_str(role="user", content=f"Here is my profile:\n\n'''\n{candidate_profile}\n'''")]
    matches = {}
    for job in jobs_list:
        messages = base_messages.copy()
        messages.append(ChatMessage.from_str(role="user", content=f"And here is the JSON card of a job that I found:\n\n'''\n{json.dumps(job)}\n'''\n\nCan you evaluate the match for me?"))
        response = await llm_struct.achat(messages)
        json_response = json.loads(response.message.blocks[0].text)
        matches.update({f"{job['job_title']} at {job['company']} ({job['url']})": {"score": json_response['match_score'], "reasons": json_response['reasons']}})
    return json.dumps(matches)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server_type", type=str, default="sse", choices=["sse", "stdio"]
    )
    args = parser.parse_args()
    mcp.run(args.server_type)