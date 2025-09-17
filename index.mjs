import { AIMessage } from '@langchain/core/messages';
import { ChatYandexGPT } from '@langchain/yandex';
import { zodToJsonSchema } from 'zod-to-json-schema';

export class LangChainYandexGPT extends ChatYandexGPT {
  static parseChatHistory(history) {
    const chatHistory = [];

    for (const message of history) {
      if (typeof message.content !== 'string') {
        throw new Error(
          'ChatYandexGPT does not support non-string message content.'
        );
      }
      if ('content' in message) {
        switch (message._getType()) {
          case 'human': {
            chatHistory.push({role: 'user', text: message.content});
            break;
          }
          case 'tool': {
            chatHistory.push({
              role: 'user',
              toolResultList: {
                toolResults: [{
                  'functionResult': {
                    'name': message.additional_kwargs.name,
                    'content': message.content,
                  },
                }],
              }});
            break;
          }
          case 'ai': {
            chatHistory.push({
              role: 'assistant',
              'toolCallList': {
                'toolCalls': message.tool_calls.map(t => {
                  return {
                    functionCall: {
                      name: t.name,
                      arguments: t.args,
                    },
                  };
                }),
              }
            });
            break;
          }
          case 'system': {
            chatHistory.push({role: 'system', text: message.content});
            break;
          }
          default: {
            console.warn('YANDEX GPT Unknown type:', message._getType());
            break;
          }
        }
      }
    }
    return chatHistory;
  }
  async completion(body, options) {
    const YANDEX_LLM_API_URL = 'https://llm.api.cloud.yandex.net/foundationModels/v1/completion';
    const response = await fetch(YANDEX_LLM_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Api-Key ${this.apiKey}`,
      },
      body: JSON.stringify(body),
      signal: options?.signal,
    });
    if (!response.ok) {
      throw new Error(`Yandex LLM completion error: ${response.status} - ${response.statusText}`);
    }
    return response.json();
  }
  async asyncCompletion(body, options) {
    const YANDEX_LLM_API_URL = 'https://llm.api.cloud.yandex.net/foundationModels/v1/completionAsync';
    const response = await fetch(YANDEX_LLM_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Api-Key ${this.apiKey}`,
      },
      body: JSON.stringify(body),
      signal: options?.signal,
    });
    if (!response.ok) {
      throw new Error(`Yandex LLM async completion error: ${response.status} - ${response.statusText}`);
    }
    return await response.json();
  }
  async checkStatus(id, options) {
    const TIMEOUT = 1000;
    return new Promise(async (resolve, reject) => {
      try {
        const response = await fetch('https://operation.api.cloud.yandex.net/operations/' + id, {
          method: 'GET',
          headers: {
            'Authorization': `Api-Key ${this.apiKey}`,
          },
          signal: options?.signal,
        });
        const result = await response.json();
        if (result.done) {
          return resolve(result);
        }
        setTimeout(() => {
          return this.checkStatus(id)
            .then(resolve)
            .catch(reject);
        }, TIMEOUT);
      } catch (error) {
        reject(error);
      }
    });
  }
  bindTools(tools) {
    this._tools = tools;
    return this;
  }
  get tools() {
    return Array.from(this._tools).map(t => {
      const jsonSchema = zodToJsonSchema(t.schema, 'parameters');
      return {
        function: {
          name: t.name,
          description: t.description,
          parameters: jsonSchema.definitions.parameters,
        }
      };
    })
  }
  async _generate(messages, options) {
    const params = {
      modelUri: this.modelURI,
      tools: this.tools,
      completionOptions: {
        temperature: this.temperature,
        maxTokens: this.maxTokens,
        stream: false,
        reasoningOptions: {
          mode: 'DISABLED',
        },
      },
      messages: LangChainYandexGPT.parseChatHistory(messages),
    };
    const { result } = await this.completion(params, options);
    const generations = [];

    switch (result.alternatives[0].status) {
      case 'ALTERNATIVE_STATUS_FINAL': {
        generations.push({
          text: result.alternatives[0].message.text,
          message: new AIMessage(result.alternatives[0].message.text),
        });
        break;
      }
      case 'ALTERNATIVE_STATUS_TOOL_CALLS': {
        generations.push({
          message: new AIMessage({
            // 'id': '', // todo - поддержать
            'content': '',
            'additional_kwargs': {
              // "tool_calls": [ // todo - поддержать
              //   {
              //     "id": "",
              //     "type": "function",
              //     "function": {
              //       "name": 't.functionCall.name',
              //       "arguments": ...
              //     }
              //   }
              // ],
            },
            'tool_calls': result.alternatives[0].message.toolCallList.toolCalls.map(t => {
              return {
                'name': t.functionCall.name,
                'args': t.functionCall.arguments,
                'type': 'tool_call',
                // 'id': '' // todo - поддержать
              };
            }),
            'invalid_tool_calls': [],
          }),
        });
        break;
      }
      default: {
        console.warn('YandexGPT returns unknown status:', result.alternatives[0].status);
        break;
      }
    }
    const { totalTokens, completionTokens, inputTextTokens } = result.usage;

    return {
      generations,
      llmOutput: {
        tokenUsage: {
          completionTokens: Number(completionTokens),
          promptTokens: Number(inputTextTokens),
          totalTokens: Number(totalTokens),
        },
        // "finish_reason": "tool_calls", // todo - поддержать
        // "system_fingerprint": "" // todo - поддержать
      },
    };
  }
}
