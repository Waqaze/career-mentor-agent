from agents import Agent,Runner,OpenAIChatCompletionsModel,AsyncOpenAI
from agents.run import RunConfig
from openai import AsyncOpenAI
from dotenv import load_dotenv
import asyncio
import os
import chainlit as cl
from agents.handoffs import Handoff

load_dotenv()

async def main():
    
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    externalclient = AsyncOpenAI(
        api_key = GEMINI_API_KEY,
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    model = OpenAIChatCompletionsModel(
        model = "gemini-2.0-flash",
        openai_client = externalclient   
    )
    config = RunConfig(
        model_provider = externalclient,
        model = model,
        tracing_disabled = True
    )
    roadmap_agent = Agent(  
        name = "Roadmap_Agent",
        instructions = "You are a roadmap maker. You help the user create a career roadmap based on their interests and skills.",
    )
    career_agent = Agent(
        name = "Career Agent",
        instructions = "You are a career advisor. You suggest fields to user and give them a career roadmap and you have tool to create a roadmap of that field which user asks.",
        tools = [
            roadmap_agent.as_tool(
                tool_name = "Roadmap-Agent",
                tool_description = "Ask user to create roadmap of that field which user's ask for. You can ask user for their interests and skills to create a roadmap.",
            )
            ],
        model = model,
    ) 
    skills_agent = Agent(
        name = "Skills Agent", 
        instructions = "You are a skills advisor. Help the user with their skills questions.",
        model = model,
    )
    job_agent = Agent(
        name = "Job Agent",
        instructions = "You are a job advisor. Help the user with their job questions.",
        model = model,
    )
    agent = Agent(
        name = "Triage Agent",
        instructions = """you are a triage agent,You deligate the user's questions to the appropriate agent based on the topic. If the question is about career, delegate to the Career Agent. If it is about skills, delegate to the Skills Agent. If it is about jobs, delegate to the Job Agent,if answer is not related to any agent polietly decline to answer.""",
        handoffs = [
            career_agent,
            skills_agent,
            job_agent,
        ],
        model = model,
    )

    @cl.on_chat_start
    async def on_chat_start():
        cl.user_session.set("history",[])
    
    @cl.on_message
    async def on_message(message: cl.Message):
        history = cl.user_session.get("history",[])
        history.append({"role":"user","content":message.content})
        result = await Runner.run(
            starting_agent = agent,
            input = history,
            run_config = config
        )
        history.append({"role":"assistant","content":result.final_output})
        await cl.Message(result.final_output).send()
        # print(history)
        print(result)

asyncio.run(main())