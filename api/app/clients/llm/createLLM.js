const { ChatOpenAI } = require('langchain/chat_models/openai');
const { sanitizeModelName } = require('../../../utils');
const { isEnabled } = require('../../../server/utils');

/**
 * Creates a new instance of a language model (LLM) for chat interactions.
 *
 * @param {Object} options - The options for creating the LLM.
 * @param {ModelOptions} options.modelOptions - The options specific to the model, including modelName, temperature, presence_penalty, frequency_penalty, and other model-related settings.
 * @param {ConfigOptions} options.configOptions - Configuration options for the API requests, including proxy settings and custom headers.
 * @param {Callbacks} options.callbacks - Callback functions for managing the lifecycle of the LLM, including token buffers, context, and initial message count.
 * @param {boolean} [options.streaming=false] - Determines if the LLM should operate in streaming mode.
 * @param {string} options.openAIApiKey - The API key for OpenAI, used for authentication.
 * @param {AzureOptions} [options.azure={}] - Optional Azure-specific configurations. If provided, Azure configurations take precedence over OpenAI configurations.
 *
 * @returns {ChatOpenAI} An instance of the ChatOpenAI class, configured with the provided options.
 *
 * @example
 * const llm = createLLM({
 *   modelOptions: { modelName: 'gpt-3.5-turbo', temperature: 0.2 },
 *   configOptions: { basePath: 'https://example.api/path' },
 *   callbacks: { onMessage: handleMessage },
 *   openAIApiKey: 'your-api-key'
 * });
 */
function createLLM({
  modelOptions,
  configOptions,
  callbacks,
  streaming = false,
  openAIApiKey,
  azure = {},
}) {
  let credentials = { openAIApiKey };
  let configuration = {
    apiKey: openAIApiKey,
  };

  let azureOptions = {};
  if (azure) {
    const useModelName = isEnabled(process.env.AZURE_USE_MODEL_AS_DEPLOYMENT_NAME);

    credentials = {};
    configuration = {};
    azureOptions = azure;

    azureOptions.azureOpenAIApiDeploymentName = useModelName
      ? sanitizeModelName(modelOptions.modelName)
      : azureOptions.azureOpenAIApiDeploymentName;
  }

  if (azure && process.env.AZURE_OPENAI_DEFAULT_MODEL) {
    modelOptions.modelName = process.env.AZURE_OPENAI_DEFAULT_MODEL;
  }
  
  // override GPT model
  // configuration.baseURL = 'http://localhost:50217/v1';
  // modelOptions.modelName = 'meetkai/functionary-small-v2.2'; //'Trelis/Llama-2-7b-chat-hf-function-calling-v3'; // meta-llama/Llama-2-7b-chat-hf';
  // // Error in applying chat template from request: Conversation roles must alternate user/assistant/user/assistant/..
  // // modelOptions.modelName = 'mistralai/Mistral-7B-Instruct-v0.2';

  const first = {
    streaming,
    verbose: true,
    credentials,
    configuration,
    ...azureOptions,
    ...modelOptions,
    ...credentials,
    callbacks,
  };
  console.log(`[createLLM] first arg to ChatOpenAI: ${JSON.stringify(first, null, 2)}`);

  return new ChatOpenAI(
    {
      streaming,
      verbose: true,
      credentials,
      configuration,
      ...azureOptions,
      ...modelOptions,
      ...credentials,
      callbacks,
    },
    configOptions,
  );
}

module.exports = createLLM;
