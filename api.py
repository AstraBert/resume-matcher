from llama_index.tools.mcp import McpToolSpec, BasicMCPClient
from llama_index.llms.groq import Groq
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent, ToolCall, ToolCallResult
from utils import ChatHistory
from fastapi import FastAPI
from llama_cloud_services import LlamaExtract
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel
import json
import gradio as gr
import requests as rq
from os import environ as ENV
from dotenv import load_dotenv

load_dotenv()

class ApiInput(BaseModel):
    resume: str

class ApiOutput(BaseModel):
    response: str
    process: str

app = FastAPI(default_response_class=ORJSONResponse)

hist = ChatHistory()
mcp_client = BasicMCPClient("http://localhost:8000/sse")
mcp_tools = McpToolSpec(mcp_client)
llm = Groq(model="qwen-qwq-32b", api_key=ENV["groq_api_key"])
extractor = LlamaExtract(api_key=ENV["llamacloud_api_key"])
extractor_agent = extractor.get_agent(id="99d18493-f573-47ba-a421-d3336260d712")

@app.post("/chat")
async def chat(inpt: ApiInput) -> ApiOutput:
    tools = await mcp_tools.to_tool_list_async()
    agent = FunctionAgent(
        llm = llm,
        name = "ResumeMatcher",
        description="Useful to match resume with jobs scraped from the web",
        system_prompt="You are the ResumeMatcher agent. Your task is to match a resume with jobs you can find from the web, evaluate the matches and return to the user a comprehensive summary of these matches, using the available tools. You should follow this workflow:\n1. Use the potential job titles contained in the resume data to perform a web search of the jobs and extract the top 5 matches within the EU, using the 'job_searcher' tool (providing it a list of potential job titles)\n2. With the information derived from step (1), pass the candidate profile (from the input resume data) and the jobs (in the same JSON string format as you got them from step (1)) to the 'evaluate_job_match' tool.\n\n3. From the job matching evaluation that you got from step (2), create a final response that summarizes the jobs and reports their match with the candidate. Don't forget to mention the company offering the job, the link to the job posting and the job title.\n\nDo not stop unless you created a final response.",
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

def add_message(history: list, message: dict):
    for x in message["files"]:
        history.append({"role": "user", "content": {"path": x}})
    if message["text"] is not None:
        history.append({"role": "user", "content": message["text"]})
    return history, gr.MultimodalTextbox(value=None, interactive=False)

def bot(history: list):
    messages = history.copy()
    messages.reverse()
    print(messages)
    msgs = [msg for msg in messages if isinstance(msg["content"], tuple)]
    if len(msgs) == 0:
        history.append({"role": "assistant", "content": "There is no attached resume"})
        return history
    else:
        resume_path = msgs[0]["content"][0]
        parsed_resume = resume_parser(resume_path)
        response = rq.post("http://localhost:7500/chat", json=ApiInput(resume=parsed_resume).model_dump())
        if response.status_code == 200:
            res_json = response.json()
            agent_process = res_json["process"]
            answer = res_json["response"]
            history.append({"role": "assistant", "content": f"## Agent process\n\n{agent_process}"})
            history.append({"role": "assistant", "content": answer})
            return history
        else:
            history.append({"role": "assistant", "content": "An error occurred while generating your response"})
            return history

with gr.Blocks(theme=gr.themes.Soft(), title="Resume Matcher") as demo:
    title = gr.HTML("<h1 align='center'>Resume Matcher</h1>\n<h2 align='center'>Match your resume with a job in the EU, effortlessly</h2>")
    chatbot = gr.Chatbot(elem_id="chatbot", bubble_full_width=False, type="messages", min_height=700, min_width=700, label="Pdf2Notes Chat", show_copy_all_button=True)

    chat_input = gr.MultimodalTextbox(
        interactive=True,
        file_count="single",
        file_types=[".pdf",".PDF", ".docx", ".doc", ".DOCX", ".DOC"],
        placeholder="Enter message or upload file...",
        show_label=False,
        sources=["upload"],
    )

    chat_msg = chat_input.submit(
        add_message, [chatbot, chat_input], [chatbot, chat_input]
    )
    bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name="bot_response")
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

app = gr.mount_gradio_app(app, demo, "/")