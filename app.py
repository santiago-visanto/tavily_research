import asyncio
from langchain_core.messages import HumanMessage
from app.taxes import app

# You may update the content of the human message with some guidelines of your own
your_additional_guidelines=f"""Analiza la situación legal y fiscal en Colombia respecto a que una empresa asuma los gastos de educación de los hijos de sus empleados. Por favor, aborda los siguientes puntos:

Legalidad: ¿Es legalmente permitido que una empresa en Colombia pague los gastos de colegio de los hijos de sus empleados?
Deducibilidad fiscal: ¿Son estos gastos deducibles de impuestos para la empresa? Si es así, ¿bajo qué condiciones?
Jurisprudencia: ¿Existen sentencias judiciales relevantes sobre este tema? Menciona las más importantes si las hay.
Implementación: Si es legal, ¿cómo se podría implementar este beneficio de manera adecuada?
Riesgos potenciales: ¿Qué riesgos legales, fiscales o laborales podría enfrentar la empresa al ofrecer este beneficio?
Vigencia de la información: ¿Hay alguna legislación reciente o pendiente que pueda afectar esta práctica en el futuro cercano?
Recomendaciones: Basándote en tu análisis, ¿qué recomendarías a una empresa que esté considerando ofrecer este beneficio a sus empleados?

Por favor, proporciona una respuesta detallada, citando leyes, regulaciones o fuentes relevantes cuando sea posible."""
messages = [
    HumanMessage(content=your_additional_guidelines)
]

async def main():
    async for s in app.astream({"messages":messages}, stream_mode="values"):
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

if __name__ == "__main__":
    asyncio.run(main())