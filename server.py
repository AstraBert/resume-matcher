from mcp.server.fastmcp import FastMCP
import datetime
from pydantic import BaseModel, Field
from typing import List
from llama_index.core.llms import ChatMessage
import json
from llama_index.llms.groq import Groq
import argparse
from typing import Literal
from linkup import LinkupClient

class JobDescription(BaseModel):
    job_title: str = Field(description="Job Title sponsored in the job announcement")
    experience_level: Literal["internship", "entry level", "junior", "mid-level", "senior"] = Field(description="Required experience level")
    required_skills: List[str] = Field(description="List of required skilled for the job")
    remote: bool = Field(description="Whether the job is remote or not")
    location: str | None = Field(description="Location, if there is any location restriction in the job")
    salary: int | None = Field(description="Yearly salary, when available")
    job_post_url: str = Field(description="URL to the job announcement")
    company: str = Field(description="Company hiring for the job")

class JobAnnouncements(BaseModel):
    jobs: List[JobDescription]

class JobMatchEvaluation(BaseModel):
    match_score: int = Field(description="An evaluation, between 0 and 100, of how much the job details match the resume data from the candidate")
    reasons: str = Field(description="Reasons for the evaluation")

with open("/run/secrets/linkup_key") as f:
    linkup_api_key = f.read()
f.close()

with open("/run/secrets/groq_key") as g:
    groq_api_key = g.read()
g.close()

mcp = FastMCP(name = "Resume Matcher MCP")
linkup_client = LinkupClient(api_key=linkup_api_key)
llm = Groq(model="qwen-qwq-32b", api_key=groq_api_key)
llm_struct = llm.as_structured_llm(JobMatchEvaluation)

@mcp.tool(name="job_searcher", description="Search for a job with a given title and location. Requires: job_title (List[str]) - the job titles you are searching for, it's a list with one or more elements.")
def job_searcher(job_description: str):
    today = datetime.datetime.now()
    date = today - datetime.timedelta(days=7)
    real_date = datetime.date(date.year, date.month, date.day)
    search_outcome = linkup_client.search(query=job_description, depth="standard", output_type="structured", include_images=False, structured_output_schema=JobAnnouncements, from_date=real_date)
    return search_outcome.model_dump_json(indent=4)
    
@mcp.tool(name="evaluate_job_match", description = "Evaluates the match between jobs opening and the candidate's profile. Requires as input: candidate_profile (str) - the data acquired from the candidate's resume with the 'resume_parser' tool, jobs (str): a JSON-like string of the job openings, retrieved from a job search with the 'job_searcher' tool")
async def evaluate_job_match(candidate_profile: str, jobs: str):
    jobs_list = json.loads(jobs)
    print(jobs_list, flush=True)
    base_messages = [ChatMessage.from_str(role="system", content="You are a job matching assistant. Your task is to evaluate a job based on its match with the candidate's profile, taking into account the job title, the skills required, the seniority level, the physical location (where the company offering the work is based in) and the working location (remote/hybrid/on-site). You then have to produce a match score (between 0 and 100) and justify that match scores explaining your reasons for that."), ChatMessage.from_str(role="user", content=f"Here is my profile:\n\n'''\n{candidate_profile}\n'''")]
    matches = {}
    for job in jobs_list:
        messages = base_messages.copy()
        messages.append(ChatMessage.from_str(role="user", content=f"And here is the JSON card of a job that I found:\n\n'''\n{json.dumps(job)}\n'''\n\nCan you evaluate the match for me?"))
        response = await llm_struct.achat(messages)
        json_response = json.loads(response.message.blocks[0].text)
        matches.update({f"{job['job_title']} at {job['company']} ({job['job_post_url']})": {"score": json_response['match_score'], "reasons": json_response['reasons']}})
        print({f"{job['job_title']} at {job['company']} ({job['job_post_url']})": {"score": json_response['match_score'], "reasons": json_response['reasons']}}, flush = True)
    return json.dumps(matches)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server_type", type=str, default="sse", choices=["sse", "stdio"]
    )
    args = parser.parse_args()
    mcp.run(args.server_type)