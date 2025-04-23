from llama_index.tools.mcp import McpToolSpec, BasicMCPClient
from llama_index.llms.groq import Groq
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent, ToolCall, ToolCallResult
from utils import ChatHistory
import redis.asyncio as redis
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from llama_cloud_services import LlamaExtract
from auth import authenticate_user
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel
import json
import gradio as gr
import requests as rq

class ApiInput(BaseModel):
    resume: str

class ApiOutput(BaseModel):
    response: str
    process: str

with open("/run/secrets/internal_key", "r") as f:
    internal_key = f.read()
f.close()

@asynccontextmanager
async def lifespan(_: FastAPI):
    redis_connection = redis.from_url("redis://resume_matcher_redis:6379", encoding="utf8")
    await FastAPILimiter.init(redis_connection)
    yield
    await FastAPILimiter.close()

app = FastAPI(default_response_class=ORJSONResponse, lifespan=lifespan)

async def check_api_key(x_api_key: str = Header(None)):
    if x_api_key == internal_key:
        return x_api_key
    else:
        raise HTTPException(status_code=401, detail="Invalid API key")

with open("/run/secrets/llamacloud_key") as f:
    llamacloud_api_key = f.read()
f.close()

with open("/run/secrets/groq_key") as g:
    groq_api_key = g.read()
g.close()

hist = ChatHistory()
mcp_client = BasicMCPClient("http://resume_matcher_mcp_server:8000/sse")
mcp_tools = McpToolSpec(mcp_client)
llm = Groq(model="llama-3.3-70b-versatile", api_key=groq_api_key)
extractor = LlamaExtract(api_key=llamacloud_api_key)
extractor_agent = extractor.get_agent(name="resume-parser")

@app.post("/chat", dependencies=[Depends(RateLimiter(times=10, seconds=60))])
async def chat(inpt: ApiInput, x_api_key: str = Depends(check_api_key)) -> ApiOutput:
    tools = await mcp_tools.to_tool_list_async()
    agent = FunctionAgent(
        llm = llm,
        name = "ResumeMatcher",
        description="Useful to match resume with jobs scraped from the web",
        system_prompt="You are the ResumeMatcher agent. Your task is to match a resume with jobs you can find from the web, evaluate the matches and return to the user a comprehensive summary of these matches, using the available tools. You should follow this workflow:\n1. Starting from the candidate description deriving from the resume, transform it into a job searching query to retrieve the top matching jobs that fit the candidate profile, using the 'job_searcher' tool\n2. With the information derived from step (1), pass the candidate profile (from the input resume data) and the jobs (in the same JSON string format as you got them from step (1)) to the 'evaluate_job_match' tool.\n\n3. From the job matching evaluation that you got from step (2), create a final response that summarizes the jobs and reports their match with the candidate. Don't forget to mention the company offering the job, the link to the job posting and the job title.\n\nDo not stop unless you completed step (1) and (2) and you created a final response.",
        tools = tools
    )
    workflow = AgentWorkflow(
        agents = [agent],
        root_agent = agent.name
    )
    process = ""
    handler = workflow.run(user_msg=f"Path to resume: {inpt.resume}", chat_history=hist.get_history())
    async for event in handler.stream_events():
        if isinstance(event, ToolCall):
            process += f"Calling tool **{event.tool_name}** with arguments:\n```json\n{json.dumps(event.tool_kwargs, indent = 4)}\n```\n\n"
        elif isinstance(event, ToolCallResult):
            process += f"Results from tool **{event.tool_name}**:\n{event.tool_output}\n\n"
        else:
            continue
    response = await handler
    response = str(response)
    return ApiOutput(response = response, process = process)

def resume_parser(path_to_resume: str):
    response = extractor_agent.extract(path_to_resume)
    extracted_data = response.data
    formatted_data = f"""
    Potential Job Roles: {', '.join(extracted_data['potential_job_titles'])}
    Seniority: {extracted_data['seniority']}
    Skills: {', '.join(extracted_data['skills'])}
    Based in: {extracted_data['based_in'] if extracted_data['based_in'] is not None else 'Information not available'}
    Working location: {extracted_data['work_location'] if extracted_data['work_location'] is not None else 'Information not available'}
    """
    return formatted_data

def bot(resume_path: str):
    headers = {"Content-Type": "application/json", "x-api-key": internal_key}
    parsed_resume = resume_parser(resume_path)
    response = rq.post("http://localhost:80/chat", json=ApiInput(resume=parsed_resume).model_dump(), headers=headers)
    if response.status_code == 200:
        res_json = response.json()
        agent_process = res_json["process"]
        answer = res_json["response"]
        return f"<details>\n\t<summary><b>Agentic Process</b></summary>\n\n{agent_process}\n\n</details>\n\n" + answer
    else:
        return "An error occurred while generating your response. Please feel free to report any error to [GitHub Discussions](https://github.com/AstraBert/resume-matcher/discussions)."

with gr.Blocks(theme=gr.themes.Soft(), title="Match-Your-Resume") as demo:
    title = gr.HTML("<h2 align='center'>Match your resume with a job, effortlessly</h2>")
    with gr.Row():
        with gr.Column():
            chat_input = gr.File(label="Upload your resume here", file_count="single", file_types=[".pdf", ".PDF", ".docx", ".DOCX", ".doc", ".DOC"])
            md_output = gr.Markdown(label="Matches", container=True, value="### No resume uploaded yet", show_label=True, show_copy_button=True)
            btn = gr.Button("Match your resume!⚗️").click(fn=bot, inputs=[chat_input], outputs=[md_output])

with gr.Blocks() as donation:
    gr.HTML("""<h2 align="center">If you find Match-Your-Resume useful, please consider to support us through donation:</h2>
<div align="center">
    <a href="https://github.com/sponsors/AstraBert"><img src="https://img.shields.io/badge/sponsor-30363D?style=for-the-badge&logo=GitHub-Sponsors&logoColor=#EA4AAA" alt="GitHub Sponsors Badge"></a>
</div>""")
    gr.HTML("<h3 align='center'>Your donation is crucial to keep this project open source and free for everyone, forever: thanks in advance!🙏</h3>")
    gr.HTML("<br>")
    gr.HTML("""<div align='center'>
        <img src="https://pnjuuftbupelnuqgkyko.supabase.co/storage/v1/object/sign/image/024b6975-855f-4ea1-8602-165a5020c3c1.png?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6InN0b3JhZ2UtdXJsLXNpZ25pbmcta2V5XzA1ZDUyOWFhLTU3YzktNDBhYS1hMDczLWI0OGYwNmI3YTAxMSJ9.eyJ1cmwiOiJpbWFnZS8wMjRiNjk3NS04NTVmLTRlYTEtODYwMi0xNjVhNTAyMGMzYzEucG5nIiwiaWF0IjoxNzQ1MzIyNDY2LCJleHAiOjIwNjA2ODI0NjZ9.GJ4pnYCB9p37WvONhyA20307PbJRo8tGdYI48NVdkKg" alt="Match-Your-Resume">
</div>""")

iface = gr.TabbedInterface(interface_list=[donation, demo], tab_names=["Home Page🏠", "Match your resume!💼"],theme=gr.themes.Soft(), title="Match-Your-Resume")

app = gr.mount_gradio_app(app, iface, "", auth=authenticate_user, auth_message="Input your username and password. If you are not already registered, go to <a href='https://register.match-your-resume.fyi'><u>the registration page</u></a>.<br><u><a href='https://register.match-your-resume.fyi'>Forgot your password?</a></u>")