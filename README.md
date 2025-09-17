# LangChain YandexGPT

LangChain-YandexGPT provides an integration of YandexGPT with LangChain for Node.js, enabling text generation and tool calling workflows. It supports API key and folder ID authentication, model selection such as yandexgpt-lite, temperature control, and structured tool invocation with zod schemas, as demonstrated below.

## Installation

```bash
npm install langchain-yandexgpt
```

## Example with tool:

```js
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { tool } from '@langchain/core/tools';
import { LangChainYandexGPT } from 'langchain-yandexgpt';
import { z } from 'zod';

const llm = new LangChainYandexGPT({
  temperature: 0,
  apiKey: YANDEX_GPT_API_KEY,
  folderID: YANDEX_GPT_CATALOG,
  model: 'yandexgpt-lite',
});

const weatherTool = tool(
  async ({ city }) => {
    return `Погода в городе ${city}: солнечно, температура 22°C, влажность 65%, легкий ветер 5 м/с`;
  },
  {
    name: 'get_weather',
    description: 'Получает актуальную информацию о погоде в указанном городе',
    schema: z.object({
      city: z.string().describe('Название города, например "Москва" или "Санкт-Петербург"'),
    }),
  }
);

const modelWithTools = llm.bindTools([weatherTool]);

const response = await modelWithTools.invoke([
  new SystemMessage(
      'Используй инструмент get_weather чтобы получить информацию о погоде',
  ),
  new HumanMessage('Какая погода в Москве?'),
]);

if (response.tool_calls && response.tool_calls.length > 0) {
  const toolResults = [];
  for (const toolCall of response.tool_calls) {
    if (toolCall.name === 'get_weather') {
      const result = await weatherTool.invoke(toolCall.args);
      console.log(`Результат инструмента: ${toolCall.name} с аргументами: ${toolCall.args}:`, result);
    }
  }
}
```
