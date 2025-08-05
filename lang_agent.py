from agents import Agent, Runner
import asyncio
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

spanish_agent = Agent(
    name="Spanish agent",
    instructions="You only speak Spanish.",
    model="gpt-4o-mini"
)

english_agent = Agent(
    name="English agent",
    instructions="You only speak English",
    model="gpt-4o-mini"
)

triage_agent = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[spanish_agent, english_agent],
    model="gpt-4o-mini"
)


async def main():
    result = await Runner.run(triage_agent, input="Hola, ¿cómo estás?")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())