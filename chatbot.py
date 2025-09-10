import chainlit as cl
from agents import Agent, Runner, set_tracing_disabled, ModelSettings
from openai.types.responses import ResponseTextDeltaEvent
from agents.extensions.models.litellm_model import LitellmModel

set_tracing_disabled(True)

api_key = "not_needed_for_local_LLM"

model_name = "huihuillama3.1-8k" # plug in model name here, that is currently run locally

agent = Agent(name="Assistant", instructions="You are a helpful assistant, Keep answers short",
              model=LitellmModel(model=f"ollama/{model_name}", api_key=api_key),
              model_settings=ModelSettings(temperature=0.7))

@cl.on_chat_start
async def chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content=f"Hi Welcome to Your Own Local Chatbot").send()


@cl.on_message
async def handle_msg(message: cl.Message):
    history = cl.user_session.get("history")
    
    msg = cl.Message(content="")
    await msg.send()

    history.append({"role": "user", "content": message.content})
    result = Runner.run_streamed(
        agent,
        input=history
    )

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)

    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)
