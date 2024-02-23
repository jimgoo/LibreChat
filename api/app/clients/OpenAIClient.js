const OpenAI = require('openai');
const { HttpsProxyAgent } = require('https-proxy-agent');
const { getResponseSender } = require('librechat-data-provider');
const { encoding_for_model: encodingForModel, get_encoding: getEncoding } = require('tiktoken');
const { encodeAndFormat, validateVisionModel } = require('~/server/services/Files/images');
const { getModelMaxTokens, genAzureChatCompletion, extractBaseURL } = require('~/utils');
const { truncateText, formatMessage, CUT_OFF_PROMPT } = require('./prompts');
const { handleOpenAIErrors } = require('./tools/util');
const spendTokens = require('~/models/spendTokens');
const { createLLM, RunManager } = require('./llm');
const { isEnabled } = require('~/server/utils');
const ChatGPTClient = require('./ChatGPTClient');
const { summaryBuffer } = require('./memory');
const { runTitleChain } = require('./chains');
const { tokenSplit } = require('./document');
const BaseClient = require('./BaseClient');
const { logger } = require('~/config');
const axios = require('axios');

// Cache to store Tiktoken instances
const tokenizersCache = {};
// Counter for keeping track of the number of tokenizer calls
let tokenizerCallsCount = 0;

class OpenAIClient extends BaseClient {
  constructor(apiKey, options = {}) {
    super(apiKey, options);
    this.ChatGPTClient = new ChatGPTClient();
    this.buildPrompt = this.ChatGPTClient.buildPrompt.bind(this);
    this.getCompletion = this.ChatGPTClient.getCompletion.bind(this);
    this.contextStrategy = options.contextStrategy
      ? options.contextStrategy.toLowerCase()
      : 'discard';
    this.shouldSummarize = this.contextStrategy === 'summarize';
    this.azure = options.azure || false;
    this.setOptions(options);
    this.promptTemplate = options.req.body.promptTemplate;
    console.log('options.req.body', options.req.body);
    console.log(`OpenAIClient.constructor: ${this.promptTemplate}`);
  }

  // TODO: PluginsClient calls this 3x, unneeded
  setOptions(options) {
    if (this.options && !this.options.replaceOptions) {
      this.options.modelOptions = {
        ...this.options.modelOptions,
        ...options.modelOptions,
      };
      delete options.modelOptions;
      this.options = {
        ...this.options,
        ...options,
      };
    } else {
      this.options = options;
    }

    if (this.options.openaiApiKey) {
      this.apiKey = this.options.openaiApiKey;
    }

    const modelOptions = this.options.modelOptions || {};

    if (!this.modelOptions) {
      this.modelOptions = {
        ...modelOptions,
        model: modelOptions.model || 'gpt-3.5-turbo',
        temperature:
          typeof modelOptions.temperature === 'undefined' ? 0.8 : modelOptions.temperature,
        top_p: typeof modelOptions.top_p === 'undefined' ? 1 : modelOptions.top_p,
        presence_penalty:
          typeof modelOptions.presence_penalty === 'undefined' ? 1 : modelOptions.presence_penalty,
        stop: modelOptions.stop,
      };
    } else {
      // Update the modelOptions if it already exists
      this.modelOptions = {
        ...this.modelOptions,
        ...modelOptions,
      };
    }

    this.isVisionModel = validateVisionModel(this.modelOptions.model);

    if (this.options.attachments && !this.isVisionModel) {
      this.modelOptions.model = 'gpt-4-vision-preview';
      this.isVisionModel = true;
    }

    if (this.isVisionModel) {
      delete this.modelOptions.stop;
    }

    const { OPENROUTER_API_KEY, OPENAI_FORCE_PROMPT } = process.env ?? {};
    if (OPENROUTER_API_KEY && !this.azure) {
      this.apiKey = OPENROUTER_API_KEY;
      this.useOpenRouter = true;
    }

    const { reverseProxyUrl: reverseProxy } = this.options;

    if (
      !this.useOpenRouter &&
      reverseProxy &&
      reverseProxy.includes('https://openrouter.ai/api/v1')
    ) {
      this.useOpenRouter = true;
    }

    this.FORCE_PROMPT =
      isEnabled(OPENAI_FORCE_PROMPT) ||
      (reverseProxy && reverseProxy.includes('completions') && !reverseProxy.includes('chat'));

    if (typeof this.options.forcePrompt === 'boolean') {
      this.FORCE_PROMPT = this.options.forcePrompt;
    }

    if (this.azure && process.env.AZURE_OPENAI_DEFAULT_MODEL) {
      this.azureEndpoint = genAzureChatCompletion(this.azure, this.modelOptions.model);
      this.modelOptions.model = process.env.AZURE_OPENAI_DEFAULT_MODEL;
    } else if (this.azure) {
      this.azureEndpoint = genAzureChatCompletion(this.azure, this.modelOptions.model);
    }

    const { model } = this.modelOptions;

    this.isChatCompletion = this.useOpenRouter || !!reverseProxy || model.includes('gpt');
    this.isChatGptModel = this.isChatCompletion;
    if (
      model.includes('text-davinci') ||
      model.includes('gpt-3.5-turbo-instruct') ||
      this.FORCE_PROMPT
    ) {
      this.isChatCompletion = false;
      this.isChatGptModel = false;
    }
    const { isChatGptModel } = this;
    this.isUnofficialChatGptModel =
      model.startsWith('text-chat') || model.startsWith('text-davinci-002-render');
    this.maxContextTokens = getModelMaxTokens(model) ?? 4095; // 1 less than maximum

    if (this.shouldSummarize) {
      this.maxContextTokens = Math.floor(this.maxContextTokens / 2);
    }

    if (this.options.debug) {
      logger.debug('[OpenAIClient] maxContextTokens', this.maxContextTokens);
    }

    this.maxResponseTokens = this.modelOptions.max_tokens || 1024;
    this.maxPromptTokens =
      this.options.maxPromptTokens || this.maxContextTokens - this.maxResponseTokens;

    if (this.maxPromptTokens + this.maxResponseTokens > this.maxContextTokens) {
      throw new Error(
        `maxPromptTokens + max_tokens (${this.maxPromptTokens} + ${this.maxResponseTokens} = ${
          this.maxPromptTokens + this.maxResponseTokens
        }) must be less than or equal to maxContextTokens (${this.maxContextTokens})`,
      );
    }

    this.sender =
      this.options.sender ??
      getResponseSender({
        model: this.modelOptions.model,
        endpoint: this.options.endpoint,
        endpointType: this.options.endpointType,
        chatGptLabel: this.options.chatGptLabel,
        modelDisplayLabel: this.options.modelDisplayLabel,
      });

    this.userLabel = this.options.userLabel || 'User';
    this.chatGptLabel = this.options.chatGptLabel || 'Assistant';

    this.setupTokens();

    if (!this.modelOptions.stop && !this.isVisionModel) {
      const stopTokens = [this.startToken];
      if (this.endToken && this.endToken !== this.startToken) {
        stopTokens.push(this.endToken);
      }
      stopTokens.push(`\n${this.userLabel}:`);
      stopTokens.push('<|diff_marker|>');
      this.modelOptions.stop = stopTokens;
    }

    if (reverseProxy) {
      this.completionsUrl = reverseProxy;
      this.langchainProxy = extractBaseURL(reverseProxy);
    } else if (isChatGptModel) {
      this.completionsUrl = 'https://api.openai.com/v1/chat/completions';
    } else {
      this.completionsUrl = 'https://api.openai.com/v1/completions';
    }

    if (this.azureEndpoint) {
      this.completionsUrl = this.azureEndpoint;
    }

    if (this.azureEndpoint && this.options.debug) {
      logger.debug('Using Azure endpoint');
    }

    if (this.useOpenRouter) {
      this.completionsUrl = 'https://openrouter.ai/api/v1/chat/completions';
    }

    console.log(`OpenAIClient.setOptions: completionsUrl: ${this.completionsUrl}`);

    return this;
  }

  setupTokens() {
    if (this.isChatCompletion) {
      this.startToken = '||>';
      this.endToken = '';
    } else if (this.isUnofficialChatGptModel) {
      this.startToken = '<|im_start|>';
      this.endToken = '<|im_end|>';
    } else {
      this.startToken = '||>';
      this.endToken = '';
    }
  }

  // Selects an appropriate tokenizer based on the current configuration of the client instance.
  // It takes into account factors such as whether it's a chat completion, an unofficial chat GPT model, etc.
  selectTokenizer() {
    let tokenizer;
    this.encoding = 'text-davinci-003';
    if (this.isChatCompletion) {
      this.encoding = 'cl100k_base';
      tokenizer = this.constructor.getTokenizer(this.encoding);
    } else if (this.isUnofficialChatGptModel) {
      const extendSpecialTokens = {
        '<|im_start|>': 100264,
        '<|im_end|>': 100265,
      };
      tokenizer = this.constructor.getTokenizer(this.encoding, true, extendSpecialTokens);
    } else {
      try {
        const { model } = this.modelOptions;
        this.encoding = model.includes('instruct') ? 'text-davinci-003' : model;
        tokenizer = this.constructor.getTokenizer(this.encoding, true);
      } catch {
        tokenizer = this.constructor.getTokenizer('text-davinci-003', true);
      }
    }

    return tokenizer;
  }

  // Retrieves a tokenizer either from the cache or creates a new one if one doesn't exist in the cache.
  // If a tokenizer is being created, it's also added to the cache.
  static getTokenizer(encoding, isModelName = false, extendSpecialTokens = {}) {
    let tokenizer;
    if (tokenizersCache[encoding]) {
      tokenizer = tokenizersCache[encoding];
    } else {
      if (isModelName) {
        tokenizer = encodingForModel(encoding, extendSpecialTokens);
      } else {
        tokenizer = getEncoding(encoding, extendSpecialTokens);
      }
      tokenizersCache[encoding] = tokenizer;
    }
    return tokenizer;
  }

  // Frees all encoders in the cache and resets the count.
  static freeAndResetAllEncoders() {
    try {
      Object.keys(tokenizersCache).forEach((key) => {
        if (tokenizersCache[key]) {
          tokenizersCache[key].free();
          delete tokenizersCache[key];
        }
      });
      // Reset count
      tokenizerCallsCount = 1;
    } catch (error) {
      logger.error('[OpenAIClient] Free and reset encoders error', error);
    }
  }

  // Checks if the cache of tokenizers has reached a certain size. If it has, it frees and resets all tokenizers.
  resetTokenizersIfNecessary() {
    if (tokenizerCallsCount >= 25) {
      if (this.options.debug) {
        logger.debug('[OpenAIClient] freeAndResetAllEncoders: reached 25 encodings, resetting...');
      }
      this.constructor.freeAndResetAllEncoders();
    }
    tokenizerCallsCount++;
  }

  // Returns the token count of a given text. It also checks and resets the tokenizers if necessary.
  getTokenCount(text) {
    this.resetTokenizersIfNecessary();
    try {
      const tokenizer = this.selectTokenizer();
      return tokenizer.encode(text, 'all').length;
    } catch (error) {
      this.constructor.freeAndResetAllEncoders();
      const tokenizer = this.selectTokenizer();
      return tokenizer.encode(text, 'all').length;
    }
  }

  getSaveOptions() {
    return {
      chatGptLabel: this.options.chatGptLabel,
      promptPrefix: this.options.promptPrefix,
      ...this.modelOptions,
    };
  }

  getBuildMessagesOptions(opts) {
    return {
      isChatCompletion: this.isChatCompletion,
      promptPrefix: opts.promptPrefix,
      abortController: opts.abortController,
    };
  }

  async buildMessages(
    messages,
    parentMessageId,
    { isChatCompletion = false, promptPrefix = null },
    opts,
  ) {
    let orderedMessages = this.constructor.getMessagesForConversation({
      messages,
      parentMessageId,
      summary: this.shouldSummarize,
    });
    if (!isChatCompletion) {
      return await this.buildPrompt(orderedMessages, {
        isChatGptModel: isChatCompletion,
        promptPrefix,
      });
    }

    let payload;
    let instructions;
    let tokenCountMap;
    let promptTokens;

    promptPrefix = (promptPrefix || this.options.promptPrefix || '').trim();
    if (promptPrefix) {
      promptPrefix = `Instructions:\n${promptPrefix}`;
      instructions = {
        role: 'system',
        name: 'instructions',
        content: promptPrefix,
      };

      if (this.contextStrategy) {
        instructions.tokenCount = this.getTokenCountForMessage(instructions);
      }
    }

    if (this.options.attachments) {
      const attachments = await this.options.attachments;
      const { files, image_urls } = await encodeAndFormat(
        this.options.req,
        attachments.filter((file) => file.type.includes('image')),
      );

      orderedMessages[orderedMessages.length - 1].image_urls = image_urls;
      this.options.attachments = files;
    }

    const formattedMessages = orderedMessages.map((message, i) => {
      const formattedMessage = formatMessage({
        message,
        userName: this.options?.name,
        assistantName: this.options?.chatGptLabel,
      });

      if (this.contextStrategy && !orderedMessages[i].tokenCount) {
        orderedMessages[i].tokenCount = this.getTokenCountForMessage(formattedMessage);
      }

      return formattedMessage;
    });

    // TODO: need to handle interleaving instructions better
    if (this.contextStrategy) {
      ({ payload, tokenCountMap, promptTokens, messages } = await this.handleContextStrategy({
        instructions,
        orderedMessages,
        formattedMessages,
      }));
    }

    const result = {
      prompt: payload,
      promptTokens,
      messages,
    };

    if (tokenCountMap) {
      tokenCountMap.instructions = instructions?.tokenCount;
      result.tokenCountMap = tokenCountMap;
    }

    if (promptTokens >= 0 && typeof opts?.getReqData === 'function') {
      opts.getReqData({ promptTokens });
    }

    return result;
  }

  async sendCompletion(payload, opts = {}) {

    console.log(`OpenAIClient.sendCompletion: payload: ${JSON.stringify(payload, null, 2)}`);
    console.log(`OpenAIClient.sendCompletion: opts: ${JSON.stringify(opts, null, 2)}`);

    /*
    const userToken = '';
    const url = 'https://api.chooseketamine.com/users';
    const proxyUrl = 'http://0.0.0.0:8080/';
    console.log(proxyUrl + url);
    */
    
    // await fetch(proxyUrl + url, {
    //   method: 'GET',
    //   headers: {
    //     'Content-Type': 'application/json;charset=UTF-8',
    //     'Authorization': userToken,
    //     'origin': 'https://chooseketamine.com',
    //   }
    // })
    // .then(response => response.json())
    // .then(data => console.log(data))
    // .catch(error => console.error('Error:', error));

    // const userData = await axios({
    //   url: proxyUrl + url,
    //   method: 'GET',
    //   headers: {
    //     'Content-Type': 'application/json;charset=UTF-8',
    //     'Authorization': userToken,
    //     'origin': 'https://chooseketamine.com',
    //   },
    // });
    // //console.log(`OpenAIClient.sendCompletion: userData: ${JSON.stringify(userData, null, 2)}`);
    // console.log(userData.data);

    /*
    if (this.promptTemplate === 'default') {
      const oldMessage = payload[payload.length - 1].content;
      console.log(`OpenAIClient.sendCompletion: oldMessage: ${oldMessage}`);
      
      const newMessage = `You are a helpful assistant for an at-home ketamine therapy company called Choose Your Horizon. Answer customer questions about the company, its products, and its policies.

      You'll now be given some information about a customer. You can use this information to answer any questions that the customer asks.
      
      The customer has the following attributes list:
      - Email: jimmiegoode@gmail.com
      - First name: Jimmie
      - Last name: Goode
      - Current address: 18 Beaux Rivages Dr, Shreveport, LA 71106
      - Phone number: 1112223310
      
      Choose Your Horizon products come in packs. Each pack contains a certain number of ketamine treatments.
      
      A "2 Pack" product type has two sessions. A "4 Pack" product type has four sessions. A "6 Pack" product type has six sessions. An "8 Pack" product type has eight sessions.
      
      The customer has purchased the following packs in tabular format, which includes columns for the status of each step and the confirmation link for that step if applicable.
      |   purchase_number | purchase_date   | product_type   | payment_type   | is_current_pack   | is_pack_complete   | Telemedicine Paperwork - Status   | PHQ/GAD/PCL Assessment - Status   | PHQ/GAD/PCL Assessment Followup - Status   | ID Upload - Status   | Integration Session (included w/pack) - Status   | Consultation (before 1st session for repeat customer) - Status   | 1st Session - Status   | 2nd Session - Status   | 3rd Session - Status   | 4th Session - Status   | 5th Session - Status   | 6th Session - Status   | 7th Session - Status   | 8th Session - Status   | Integration Session (add-on 1) - Status   | Integration Session (add-on 1) - Confirmation Link                                                                 | Group Integration Session (add-on 1) - Status   | Group Integration Session (add-on 1) - Confirmation Link                                                           | Consultation (after 2nd session) - Status   | Consultation (after 2nd session) - Confirmation Link                                                               | 6th Session - Confirmation Link                                                                                    | Consultation (before 1st session for new customer) - Status   | Consultation (before 1st session for new customer) - Confirmation Link                                             | 1st Session - Confirmation Link                                                                                    | 2nd Session - Confirmation Link                                                                                    |
      |------------------:|:----------------|:---------------|:---------------|:------------------|:-------------------|:----------------------------------|:----------------------------------|:-------------------------------------------|:---------------------|:-------------------------------------------------|:-----------------------------------------------------------------|:-----------------------|:-----------------------|:-----------------------|:-----------------------|:-----------------------|:-----------------------|:-----------------------|:-----------------------|:------------------------------------------|:-------------------------------------------------------------------------------------------------------------------|:------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------|:--------------------------------------------|:-------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------|
      |                 1 | 2023-06-28      | 2 Pack         | nmi            | False             | False              | completed                         | completed                         | incomplete                                 | completed            | N/A                                              | N/A                                                              | completed              | completed              | N/A                    | N/A                    | N/A                    | N/A                    | N/A                    | N/A                    | N/A                                       | N/A                                                                                                                | N/A                                             | N/A                                                                                                                | canceled                                    | https://app.acuityscheduling.com/schedule.php?owner=24101249&action=appt&id%5B%5D=ff0b8d5b22c13867ac2a215824e371ba | N/A                                                                                                                | canceled                                                      | https://app.acuityscheduling.com/schedule.php?owner=24101249&action=appt&id%5B%5D=19487f5e466a55846288a60833bc4d2f | https://app.acuityscheduling.com/schedule.php?owner=24101249&action=appt&id%5B%5D=039bef46a74f95c1fed5f4d8cda89d8f | https://app.acuityscheduling.com/schedule.php?owner=24101249&action=appt&id%5B%5D=3de23b374331d22996437978bf1a1b9d |
      |                 2 | 2023-09-11      | 4 Pack         | nmi            | False             | False              | incomplete                        | incomplete                        | incomplete                                 | incomplete           | N/A                                              | incomplete                                                       | incomplete             | incomplete             | incomplete             | incomplete             | N/A                    | N/A                    | N/A                    | N/A                    | N/A                                       | N/A                                                                                                                | N/A                                             | N/A                                                                                                                | N/A                                         | N/A                                                                                                                | N/A                                                                                                                | N/A                                                           | N/A                                                                                                                | N/A                                                                                                                | N/A                                                                                                                |
      |                 3 | 2023-10-16      | 6 Pack         | nmi            | True              | False              | completed                         | completed                         | incomplete                                 | completed            | N/A                                              | N/A                                                              | completed              | completed              | completed              | completed              | completed              | canceled               | N/A                    | N/A                    | canceled                                  | https://app.acuityscheduling.com/schedule.php?owner=24101249&action=appt&id%5B%5D=efa0a89cd8c02435be91543be5e54879 | canceled                                        | https://app.acuityscheduling.com/schedule.php?owner=24101249&action=appt&id%5B%5D=09e424d89751469b90a369d74881c6ad | completed                                   | https://app.acuityscheduling.com/schedule.php?owner=24101249&action=appt&id%5B%5D=c8c2ddddf0793e2c68f89bc825afbb73 | https://app.acuityscheduling.com/schedule.php?owner=24101249&action=appt&id%5B%5D=e6895e48fb4505ad43c459fdda3b1ca3 | N/A                                                           | N/A                                                                                                                | N/A                                                                                                                | N/A                                                                                                                |
      |                 4 | 2023-10-26      | 6 Pack         | nmi            | False             | False              | incomplete                        | incomplete                        | incomplete                                 | incomplete           | N/A                                              | incomplete                                                       | incomplete             | incomplete             | incomplete             | incomplete             | incomplete             | incomplete             | N/A                    | N/A                    | N/A                                       | N/A                                                                                                                | N/A                                             | N/A                                                                                                                | N/A                                         | N/A                                                                                                                | N/A                                                                                                                | N/A                                                           | N/A                                                                                                                | N/A                                                                                                                | N/A                                                                                                                |
      |                 5 | 2023-11-09      | 6 Pack         | nmi            | False             | False              | incomplete                        | incomplete                        | incomplete                                 | incomplete           | N/A                                              | incomplete                                                       | incomplete             | incomplete             | incomplete             | incomplete             | incomplete             | incomplete             | N/A                    | N/A                    | N/A                                       | N/A                                                                                                                | N/A                                             | N/A                                                                                                                | N/A                                         | N/A                                                                                                                | N/A                                                                                                                | N/A                                                           | N/A                                                                                                                | N/A                                                                                                                | N/A                                                                                                                |
      |                 6 | 2023-12-12      | 8 Pack         | nmi            | False             | False              | incomplete                        | incomplete                        | incomplete                                 | incomplete           | incomplete                                       | incomplete                                                       | incomplete             | incomplete             | incomplete             | incomplete             | incomplete             | incomplete             | incomplete             | incomplete             | N/A                                       | N/A                                                                                                                | N/A                                             | N/A                                                                                                                | N/A                                         | N/A                                                                                                                | N/A                                                                                                                | N/A                                                           | N/A                                                                                                                | N/A                                                                                                                | N/A                                                                                                                |
      
      The customer says: "${oldMessage}". What is your response?`;

      payload[payload.length - 1] = {role: 'user', content: newMessage};
    }
    */

    let reply = '';
    let result = null;
    let streamResult = null;
    this.modelOptions.user = this.user;
    const invalidBaseUrl = this.completionsUrl && extractBaseURL(this.completionsUrl) === null;
    const useOldMethod = !!(invalidBaseUrl || !this.isChatCompletion);
    if (typeof opts.onProgress === 'function' && useOldMethod) {
      await this.getCompletion(
        payload,
        (progressMessage) => {
          if (progressMessage === '[DONE]') {
            return;
          }

          if (progressMessage.choices) {
            streamResult = progressMessage;
          }

          let token = null;
          if (this.isChatCompletion) {
            token =
              progressMessage.choices?.[0]?.delta?.content ?? progressMessage.choices?.[0]?.text;
          } else {
            token = progressMessage.choices?.[0]?.text;
          }

          if (!token && this.useOpenRouter) {
            token = progressMessage.choices?.[0]?.message?.content;
          }
          // first event's delta content is always undefined
          if (!token) {
            return;
          }

          if (token === this.endToken) {
            return;
          }
          opts.onProgress(token);
          reply += token;
        },
        opts.abortController || new AbortController(),
      );
    } else if (typeof opts.onProgress === 'function' || this.options.useChatCompletion) {
      reply = await this.chatCompletion({
        payload,
        clientOptions: opts,
        onProgress: opts.onProgress,
        abortController: opts.abortController,
      });
    } else {
      result = await this.getCompletion(
        payload,
        null,
        opts.abortController || new AbortController(),
      );

      logger.debug('[OpenAIClient] sendCompletion: result', result);

      if (this.isChatCompletion) {
        reply = result.choices[0].message.content;
      } else {
        reply = result.choices[0].text.replace(this.endToken, '');
      }
    }

    if (streamResult && typeof opts.addMetadata === 'function') {
      const { finish_reason } = streamResult.choices[0];
      opts.addMetadata({ finish_reason });
    }
    return reply.trim();
  }

  initializeLLM({
    model = 'gpt-3.5-turbo',
    modelName,
    temperature = 0.2,
    presence_penalty = 0,
    frequency_penalty = 0,
    max_tokens,
    streaming,
    context,
    tokenBuffer,
    initialMessageCount,
  }) {
    const modelOptions = {
      modelName: modelName ?? model,
      temperature,
      presence_penalty,
      frequency_penalty,
      user: this.user,
    };

    if (max_tokens) {
      modelOptions.max_tokens = max_tokens;
    }

    const configOptions = {};

    if (this.langchainProxy) {
      configOptions.basePath = this.langchainProxy;
    }

    if (this.useOpenRouter) {
      configOptions.basePath = 'https://openrouter.ai/api/v1';
      configOptions.baseOptions = {
        headers: {
          'HTTP-Referer': 'https://librechat.ai',
          'X-Title': 'LibreChat',
        },
      };
    }

    if (this.options.proxy) {
      configOptions.httpAgent = new HttpsProxyAgent(this.options.proxy);
      configOptions.httpsAgent = new HttpsProxyAgent(this.options.proxy);
    }

    const { req, res, debug } = this.options;
    const runManager = new RunManager({ req, res, debug, abortController: this.abortController });
    this.runManager = runManager;

    const llm = createLLM({
      modelOptions,
      configOptions,
      openAIApiKey: this.apiKey,
      azure: this.azure,
      streaming,
      callbacks: runManager.createCallbacks({
        context,
        tokenBuffer,
        conversationId: this.conversationId,
        initialMessageCount,
      }),
    });

    return llm;
  }

  /**
   * Generates a concise title for a conversation based on the user's input text and response.
   * Uses either specified method or starts with the OpenAI `functions` method (using LangChain).
   * If the `functions` method fails, it falls back to the `completion` method,
   * which involves sending a chat completion request with specific instructions for title generation.
   *
   * @param {Object} params - The parameters for the conversation title generation.
   * @param {string} params.text - The user's input.
   * @param {string} [params.responseText=''] - The AI's immediate response to the user.
   *
   * @returns {Promise<string | 'New Chat'>} A promise that resolves to the generated conversation title.
   *                            In case of failure, it will return the default title, "New Chat".
   */
  async titleConvo({ text, responseText = '' }) {
    let title = 'New Chat';
    const convo = `||>User:
"${truncateText(text)}"
||>Response:
"${JSON.stringify(truncateText(responseText))}"`;

    const { OPENAI_TITLE_MODEL } = process.env ?? {};

    const model = this.options.titleModel ?? OPENAI_TITLE_MODEL ?? 'gpt-3.5-turbo';

    const modelOptions = {
      // TODO: remove the gpt fallback and make it specific to endpoint
      model,
      temperature: 0.2,
      presence_penalty: 0,
      frequency_penalty: 0,
      max_tokens: 16,
    };

    const titleChatCompletion = async () => {
      modelOptions.model = model;

      if (this.azure) {
        modelOptions.model = process.env.AZURE_OPENAI_DEFAULT_MODEL ?? modelOptions.model;
        this.azureEndpoint = genAzureChatCompletion(this.azure, modelOptions.model);
      }

      const instructionsPayload = [
        {
          role: 'system',
          content: `Detect user language and write in the same language an extremely concise title for this conversation, which you must accurately detect.
Write in the detected language. Title in 5 Words or Less. No Punctuation or Quotation. Do not mention the language. All first letters of every word should be capitalized and write the title in User Language only.

${convo}

||>Title:`,
        },
      ];

      try {
        title = (
          await this.sendPayload(instructionsPayload, { modelOptions, useChatCompletion: true })
        ).replaceAll('"', '');
      } catch (e) {
        logger.error(
          '[OpenAIClient] There was an issue generating the title with the completion method',
          e,
        );
      }
    };

    if (this.options.titleMethod === 'completion') {
      await titleChatCompletion();
      logger.debug('[OpenAIClient] Convo Title: ' + title);
      return title;
    }

    try {
      this.abortController = new AbortController();
      const llm = this.initializeLLM({ ...modelOptions, context: 'title', tokenBuffer: 150 });
      title = await runTitleChain({ llm, text, convo, signal: this.abortController.signal });
    } catch (e) {
      if (e?.message?.toLowerCase()?.includes('abort')) {
        logger.debug('[OpenAIClient] Aborted title generation');
        return;
      }
      logger.error(
        '[OpenAIClient] There was an issue generating title with LangChain, trying completion method...',
        e,
      );

      await titleChatCompletion();
    }

    logger.debug('[OpenAIClient] Convo Title: ' + title);
    return title;
  }

  async summarizeMessages({ messagesToRefine, remainingContextTokens }) {
    logger.debug('[OpenAIClient] Summarizing messages...');
    let context = messagesToRefine;
    let prompt;

    // TODO: remove the gpt fallback and make it specific to endpoint
    const { OPENAI_SUMMARY_MODEL = 'gpt-3.5-turbo' } = process.env ?? {};
    const model = this.options.summaryModel ?? OPENAI_SUMMARY_MODEL;
    const maxContextTokens = getModelMaxTokens(model) ?? 4095;

    // 3 tokens for the assistant label, and 98 for the summarizer prompt (101)
    let promptBuffer = 101;

    /*
     * Note: token counting here is to block summarization if it exceeds the spend; complete
     * accuracy is not important. Actual spend will happen after successful summarization.
     */
    const excessTokenCount = context.reduce(
      (acc, message) => acc + message.tokenCount,
      promptBuffer,
    );

    if (excessTokenCount > maxContextTokens) {
      ({ context } = await this.getMessagesWithinTokenLimit(context, maxContextTokens));
    }

    if (context.length === 0) {
      logger.debug(
        '[OpenAIClient] Summary context is empty, using latest message within token limit',
      );

      promptBuffer = 32;
      const { text, ...latestMessage } = messagesToRefine[messagesToRefine.length - 1];
      const splitText = await tokenSplit({
        text,
        chunkSize: Math.floor((maxContextTokens - promptBuffer) / 3),
      });

      const newText = `${splitText[0]}\n...[truncated]...\n${splitText[splitText.length - 1]}`;
      prompt = CUT_OFF_PROMPT;

      context = [
        formatMessage({
          message: {
            ...latestMessage,
            text: newText,
          },
          userName: this.options?.name,
          assistantName: this.options?.chatGptLabel,
        }),
      ];
    }
    // TODO: We can accurately count the tokens here before handleChatModelStart
    // by recreating the summary prompt (single message) to avoid LangChain handling

    const initialPromptTokens = this.maxContextTokens - remainingContextTokens;
    logger.debug('[OpenAIClient] initialPromptTokens', initialPromptTokens);

    const llm = this.initializeLLM({
      model,
      temperature: 0.2,
      context: 'summary',
      tokenBuffer: initialPromptTokens,
    });

    try {
      const summaryMessage = await summaryBuffer({
        llm,
        debug: this.options.debug,
        prompt,
        context,
        formatOptions: {
          userName: this.options?.name,
          assistantName: this.options?.chatGptLabel ?? this.options?.modelLabel,
        },
        previous_summary: this.previous_summary?.summary,
        signal: this.abortController.signal,
      });

      const summaryTokenCount = this.getTokenCountForMessage(summaryMessage);

      if (this.options.debug) {
        logger.debug('[OpenAIClient] summaryTokenCount', summaryTokenCount);
        logger.debug(
          `[OpenAIClient] Summarization complete: remainingContextTokens: ${remainingContextTokens}, after refining: ${
            remainingContextTokens - summaryTokenCount
          }`,
        );
      }

      return { summaryMessage, summaryTokenCount };
    } catch (e) {
      if (e?.message?.toLowerCase()?.includes('abort')) {
        logger.debug('[OpenAIClient] Aborted summarization');
        const { run, runId } = this.runManager.getRunByConversationId(this.conversationId);
        if (run && run.error) {
          const { error } = run;
          this.runManager.removeRun(runId);
          throw new Error(error);
        }
      }
      logger.error('[OpenAIClient] Error summarizing messages', e);
      return {};
    }
  }

  async recordTokenUsage({ promptTokens, completionTokens }) {
    logger.debug('[OpenAIClient] recordTokenUsage:', { promptTokens, completionTokens });
    await spendTokens(
      {
        user: this.user,
        model: this.modelOptions.model,
        context: 'message',
        conversationId: this.conversationId,
      },
      { promptTokens, completionTokens },
    );
  }

  getTokenCountForResponse(response) {
    return this.getTokenCountForMessage({
      role: 'assistant',
      content: response.text,
    });
  }

  async chatCompletion({ payload, onProgress, clientOptions, abortController = null }) {
    let error = null;
    const errorCallback = (err) => (error = err);
    let intermediateReply = '';
    try {
      if (!abortController) {
        abortController = new AbortController();
      }

      let modelOptions = { ...this.modelOptions };

      if (typeof onProgress === 'function') {
        modelOptions.stream = true;
      }
      if (this.isChatCompletion) {
        modelOptions.messages = payload;
      } else {
        // TODO: unreachable code. Need to implement completions call for non-chat models
        modelOptions.prompt = payload;
      }

      const baseURL = extractBaseURL(this.completionsUrl);
      // let { messages: _msgsToLog, ...modelOptionsToLog } = modelOptions;
      // if (modelOptionsToLog.messages) {
      //   _msgsToLog = modelOptionsToLog.messages.map((msg) => {
      //     let { content, ...rest } = msg;

      //     if (content)
      //     return { ...rest, content: truncateText(content) };
      //   });
      // }
      logger.debug('[OpenAIClient] chatCompletion', { baseURL, modelOptions });
      const opts = {
        baseURL,
      };

      if (this.useOpenRouter) {
        opts.defaultHeaders = {
          'HTTP-Referer': 'https://librechat.ai',
          'X-Title': 'LibreChat',
        };
      }

      if (this.options.headers) {
        opts.defaultHeaders = { ...opts.defaultHeaders, ...this.options.headers };
      }

      if (this.options.proxy) {
        opts.httpAgent = new HttpsProxyAgent(this.options.proxy);
      }

      if (this.isVisionModel) {
        modelOptions.max_tokens = 4000;
      }

      if (this.azure || this.options.azure) {
        // Azure does not accept `model` in the body, so we need to remove it.
        delete modelOptions.model;

        opts.baseURL = this.azureEndpoint.split('/chat')[0];
        opts.defaultQuery = { 'api-version': this.azure.azureOpenAIApiVersion };
        opts.defaultHeaders = { ...opts.defaultHeaders, 'api-key': this.apiKey };
      }

      let chatCompletion;
      const openai = new OpenAI({
        apiKey: this.apiKey,
        ...opts,
      });

      /* hacky fix for Mistral AI API not allowing a singular system message in payload */
      if (opts.baseURL.includes('https://api.mistral.ai/v1') && modelOptions.messages) {
        const { messages } = modelOptions;
        if (messages.length === 1 && messages[0].role === 'system') {
          modelOptions.messages[0].role = 'user';
        }
      }

      if (this.options.addParams && typeof this.options.addParams === 'object') {
        modelOptions = {
          ...modelOptions,
          ...this.options.addParams,
        };
      }

      if (this.options.dropParams && Array.isArray(this.options.dropParams)) {
        this.options.dropParams.forEach((param) => {
          delete modelOptions[param];
        });
      }

      let UnexpectedRoleError = false;
      if (modelOptions.stream) {
        const stream = await openai.beta.chat.completions
          .stream({
            ...modelOptions,
            stream: true,
          })
          .on('abort', () => {
            /* Do nothing here */
          })
          .on('error', (err) => {
            handleOpenAIErrors(err, errorCallback, 'stream');
          })
          .on('finalMessage', (message) => {
            if (message?.role !== 'assistant') {
              stream.messages.push({ role: 'assistant', content: intermediateReply });
              UnexpectedRoleError = true;
            }
          });

        for await (const chunk of stream) {
          const token = chunk.choices[0]?.delta?.content || '';
          intermediateReply += token;
          onProgress(token);
          if (abortController.signal.aborted) {
            stream.controller.abort();
            break;
          }
        }

        if (!UnexpectedRoleError) {
          chatCompletion = await stream.finalChatCompletion().catch((err) => {
            handleOpenAIErrors(err, errorCallback, 'finalChatCompletion');
          });
        }
      }
      // regular completion
      else {
        chatCompletion = await openai.chat.completions
          .create({
            ...modelOptions,
          })
          .catch((err) => {
            handleOpenAIErrors(err, errorCallback, 'create');
          });
      }

      if (!chatCompletion && UnexpectedRoleError) {
        throw new Error(
          'OpenAI error: Invalid final message: OpenAI expects final message to include role=assistant',
        );
      } else if (!chatCompletion && error) {
        throw new Error(error);
      } else if (!chatCompletion) {
        throw new Error('Chat completion failed');
      }

      const { message, finish_reason } = chatCompletion.choices[0];
      if (chatCompletion && typeof clientOptions.addMetadata === 'function') {
        clientOptions.addMetadata({ finish_reason });
      }

      return message.content;
    } catch (err) {
      if (
        err?.message?.includes('abort') ||
        (err instanceof OpenAI.APIError && err?.message?.includes('abort'))
      ) {
        return intermediateReply;
      }
      if (
        err?.message?.includes(
          'OpenAI error: Invalid final message: OpenAI expects final message to include role=assistant',
        ) ||
        err?.message?.includes('The server had an error processing your request') ||
        err?.message?.includes('missing finish_reason') ||
        err?.message?.includes('missing role') ||
        (err instanceof OpenAI.OpenAIError && err?.message?.includes('missing finish_reason'))
      ) {
        logger.error('[OpenAIClient] Known OpenAI error:', err);
        return intermediateReply;
      } else if (err instanceof OpenAI.APIError) {
        if (intermediateReply) {
          return intermediateReply;
        } else {
          throw err;
        }
      } else {
        logger.error('[OpenAIClient.chatCompletion] Unhandled error type', err);
        throw err;
      }
    }
  }
}

module.exports = OpenAIClient;
