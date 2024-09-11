import asyncio
from langchain_core.messages import HumanMessage
from app.company_researcher import app

# You may update the content of the human message with some guidelines of your own
company = "Tavily"
your_additional_guidelines=f"I would like a comprehensive and detailed report on the latest developments concerning the company {company}."
messages = [
    HumanMessage(content=your_additional_guidelines)
]

async def main():
    async for s in app.astream({"company": company, "messages":messages}, stream_mode="values"):
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

if __name__ == "__main__":
    asyncio.run(main())