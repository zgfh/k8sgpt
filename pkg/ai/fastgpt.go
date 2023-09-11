/*
Copyright 2023 The K8sGPT Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package ai

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"strings"

	"github.com/k8sgpt-ai/k8sgpt/pkg/cache"
	"github.com/k8sgpt-ai/k8sgpt/pkg/util"

	"github.com/sashabaranov/go-openai"

	"github.com/fatih/color"
)

type FastGptClient struct {
	client   *openai.Client
	language string
	chatId   string
	model    string
}

// type CustomChatCompletionRequest struct {
// 	openai.ChatCompletionRequest
// 	Model    string                         `json:"model"`
// 	Messages []openai.ChatCompletionMessage `json:"messages"`
// 	chatId   string
// }

func (c *FastGptClient) Configure(config IAIConfig, language string) error {
	tokenAndChatId := config.GetPassword()
	tokenAndChatIdArr := strings.Split(tokenAndChatId, ":")

	defaultConfig := openai.DefaultConfig(tokenAndChatIdArr[1])

	baseURL := config.GetBaseURL()
	if baseURL != "" {
		defaultConfig.BaseURL = baseURL
	}

	client := openai.NewClientWithConfig(defaultConfig)
	if client == nil {
		return errors.New("error creating OpenAI client")
	}
	c.language = language
	c.client = client
	c.chatId = tokenAndChatIdArr[0]
	c.model = config.GetModel()
	return nil
}

func (c *FastGptClient) GetCompletion(ctx context.Context, prompt string, promptTmpl string) (string, error) {
	// Create a completion request
	if len(promptTmpl) == 0 {
		promptTmpl = PromptMap["default"]
	}

	req := openai.ChatCompletionRequest{
		Model: c.model,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    "user",
				Content: fmt.Sprintf(promptTmpl, c.language, prompt),
			},
		},
		ChatId: c.chatId,
	}
	resp, err := c.client.CreateChatCompletion(ctx, req)
	if err != nil {
		return "", err
	}
	return resp.Choices[0].Message.Content, nil
}

func (a *FastGptClient) Parse(ctx context.Context, prompt []string, cache cache.ICache, promptTmpl string) (string, error) {
	inputKey := strings.Join(prompt, " ")
	// Check for cached data
	cacheKey := util.GetCacheKey(a.GetName(), a.language, inputKey)

	if !cache.IsCacheDisabled() && cache.Exists(cacheKey) {
		response, err := cache.Load(cacheKey)
		if err != nil {
			return "", err
		}

		if response != "" {
			output, err := base64.StdEncoding.DecodeString(response)
			if err != nil {
				color.Red("error decoding cached data: %v", err)
				return "", nil
			}
			return string(output), nil
		}
	}

	response, err := a.GetCompletion(ctx, inputKey, promptTmpl)
	if err != nil {
		return "", err
	}

	err = cache.Store(cacheKey, base64.StdEncoding.EncodeToString([]byte(response)))

	if err != nil {
		color.Red("error storing value to cache: %v", err)
		return "", nil
	}

	return response, nil
}

func (a *FastGptClient) GetName() string {
	return "fastgpt"
}
