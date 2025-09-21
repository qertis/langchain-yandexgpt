import { AIMessage } from '@langchain/core/messages';
import { ChatYandexGPT } from '@langchain/yandex';
import { zodToJsonSchema } from 'zod-to-json-schema';
import { ZodObject } from 'zod';

export class LangChainYandexGPT extends ChatYandexGPT {
  static parseChatHistory(history) {
    const chatHistory = [];

    for (const message of history) {
      if ('content' in message) {
        switch (message._getType()) {
          case 'human': {
            if (typeof message.content !== 'string') {
              throw new Error(
                'ChatYandexGPT does not support non-string message content.'
              );
            }
            const history = {
              role: 'user',
              text: message.content,
            }
            chatHistory.push(history);
            break;
          }
          case 'tool': {
            if (typeof message.content !== 'string') {
              throw new Error(
                'ChatYandexGPT does not support non-string message content.'
              );
            }
            const history = {
              role: 'assistant',
              toolResultList: {
                toolResults: [{
                  functionResult: {
                    name: message.additional_kwargs.name,
                    content: message.content,
                  },
                }],
              }
            }
            chatHistory.push(history);
            break;
          }
          case 'ai': {
            const history = {
              role: 'assistant',
              toolCallList: {},
            }
            if (message.tool_calls?.length) {
              history.toolCallList = {
                toolCalls: message.tool_calls.map(t => {
                  return {
                    functionCall: {
                      name: t.name,
                      arguments: t.args,
                    },
                  };
                }),
              }
            }
            chatHistory.push(history);
            break;
          }
          case 'system': {
            if (typeof message.content !== 'string') {
              throw new Error(
                'ChatYandexGPT does not support non-string message content.'
              );
            }
            const history = {
              role: 'system',
              text: message.content,
            }
            chatHistory.push(history);
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
  // todo: есть бесконечная рекурсия при неудачных запросах
  async checkStatus(id, options) {
    const TIMEOUT = 2000;
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
          return this.checkStatus(id, options)
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
    if (!this._tools) {
      return [];
    }
    return Array.from(this._tools).map(t => {
      if (t.schema instanceof ZodObject) {
        const jsonSchema = zodToJsonSchema(t.schema, 'parameters');
        return {
          function: {
            name: t.name,
            description: t.description,
            parameters: jsonSchema.definitions.parameters,
          }
        };
      }
      return {
        function: {
          name: t.name,
          description: t.description,
          parameters: t.schema,
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

    switch (result.alternatives[0]?.status) {
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
            additional_kwargs: {
              // tool_calls: [ // todo - поддержать
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
            tool_calls: result.alternatives[0].message.toolCallList.toolCalls.map(t => {
              return {
                name: t.functionCall.name,
                args: t.functionCall.arguments,
                type: 'tool_call',
                // 'id': '' // todo - поддержать
              };
            }),
            invalid_tool_calls: [],
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
        finish_reason: params.messages.find(m => typeof m.toolResultList === 'object') ? "tool_calls" : undefined,
        // "system_fingerprint": "" // todo - поддержать
      },
    };
  }
}
