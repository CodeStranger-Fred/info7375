package newrep

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"time"

	"github.com/CodeStranger-Fred/assignemnt1/mdp"
)

type UniformPrior struct{}

func (UniformPrior) Prior(_ mdp.State, actions []mdp.Action) map[mdp.Action]float64 {
	m := make(map[mdp.Action]float64, len(actions))
	w := 1.0 / float64(len(actions))
	for _, a := range actions {
		m[a] = w
	}
	return m
}

type OpenAIChatPrior struct {
	APIKey string
	Model  string
	Client *http.Client
}

func (p OpenAIChatPrior) Prior(s mdp.State, actions []mdp.Action) map[mdp.Action]float64 {
	key := p.APIKey
	if key == "" {
		key = os.Getenv("OPENAI_API_KEY")
	}
	if key == "" || p.Model == "" {
		return UniformPrior{}.Prior(s, actions)
	}
	payload := map[string]any{
		"model": p.Model,
		"messages": []map[string]string{
			{"role": "user", "content": fmt.Sprintf(
				"You are a grid navigation planner. Current state: %s. Goal state: G5_5. Obstacles exist at positions [S0_2, S1_2, S2_2, S3_1, S3_2, S3_3, S4_4]. "+
				"Actions=%v. Think logically which action moves closer to the goal (down or right are usually good). "+
				"Output a pure JSON mapping of action to probability that helps reach the goal, e.g. {\"up\":0.1,\"down\":0.3,\"left\":0.2,\"right\":0.4}.",
				string(s), toStringSlice(actions))},
		},
		"temperature": 0.2,
	}
	bs, _ := json.Marshal(payload)
	cli := p.Client
	if cli == nil {
		cli = &http.Client{Timeout: 12 * time.Second}
	}
	req, _ := http.NewRequest("POST", "https://api.openai.com/v1/chat/completions", bytes.NewReader(bs))
	req.Header.Set("Authorization", "Bearer "+key)
	req.Header.Set("Content-Type", "application/json")
	resp, err := cli.Do(req)
	if err != nil || resp.StatusCode/100 != 2 {
		if resp != nil {
			resp.Body.Close()
		}
		return UniformPrior{}.Prior(s, actions)
	}
	defer resp.Body.Close()
	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil || len(out.Choices) == 0 {
		return UniformPrior{}.Prior(s, actions)
	}
	var parsed map[string]float64
	if err := json.Unmarshal([]byte(out.Choices[0].Message.Content), &parsed); err != nil {
		return UniformPrior{}.Prior(s, actions)
	}
	m := make(map[mdp.Action]float64, len(actions))
	sum := 0.0
	for _, a := range actions {
		m[a] = parsed[string(a)]
		sum += m[a]
	}
	if sum <= 0 {
		return UniformPrior{}.Prior(s, actions)
	}
	for k := range m {
		m[k] /= sum
	}
	return m
}

func toStringSlice(as []mdp.Action) []string {
	o := make([]string, len(as))
	for i, a := range as {
		o[i] = string(a)
	}
	return o
}
