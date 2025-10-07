package main

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"time"

	"github.com/CodeStranger-Fred/assignemnt1/mdp"
	"github.com/CodeStranger-Fred/assignemnt1/newrep"
)

type GridWorldTransition struct {
	Width, Height int
	Goal          mdp.State
	Obstacles     map[string]bool
}

func (t GridWorldTransition) Transition(s mdp.State, a mdp.Action) mdp.ProbabilityDistribution[mdp.State] {
	var x, y int
	fmt.Sscanf(string(s), "S%d_%d", &x, &y)
	switch a {
	case "up":
		y--
	case "down":
		y++
	case "left":
		x--
	case "right":
		x++
	}
	if x < 0 {
		x = 0
	}
	if y < 0 {
		y = 0
	}
	if x >= t.Width {
		x = t.Width - 1
	}
	if y >= t.Height {
		y = t.Height - 1
	}
	ns := fmt.Sprintf("S%d_%d", x, y)
	if t.Obstacles[ns] {
		ns = string(s)
	}

	goal := string(t.Goal) 
	if strings.HasPrefix(ns, "S") && strings.HasPrefix(goal, "G") && ns[1:] == goal[1:] {
		ns = goal 
	}
	return mdp.DiscretePdf[mdp.State]{Map: map[mdp.State]mdp.Probability{
		mdp.State(ns): 1.0,
	}}
}

type GridWorldReward struct {
	Goal      mdp.State
	StepCost  float64
	GoalValue float64
}

func (r GridWorldReward) Reward(s mdp.State, a mdp.Action, ns mdp.State) mdp.ProbabilityDistribution[mdp.Reward] {
	if ns == r.Goal {
		return mdp.DiscretePdf[mdp.Reward]{Map: map[mdp.Reward]mdp.Probability{
			mdp.Reward(r.GoalValue): 1.0,
		}}
	}
	var x0, y0, x1, y1 int
	fmt.Sscanf(string(s), "S%d_%d", &x0, &y0)
	fmt.Sscanf(string(ns), "S%d_%d", &x1, &y1)
	distOld := math.Abs(float64(2-x0)) + math.Abs(float64(2-y0))
	distNew := math.Abs(float64(2-x1)) + math.Abs(float64(2-y1))
	reward := -1.0
	if distNew < distOld {
		reward += 0.5
	}
	return mdp.DiscretePdf[mdp.Reward]{Map: map[mdp.Reward]mdp.Probability{
		mdp.Reward(reward): 1.0,
	}}
}

func BuildGridWorldEnv() *mdp.MDP {
	width, height := 6, 6

	obstacles := map[string]bool{
		"S0_2": true, "S1_2": true, "S2_2": true, 
		"S3_1": true, "S3_2": true, "S3_3": true, 
		"S4_4": true, 
	}

	return &mdp.MDP{
		TransitionFunction: GridWorldTransition{
			Width:     width,
			Height:    height,
			Goal:      "G5_5",
			Obstacles: obstacles,
		},
		RewardFunction: GridWorldReward{
			Goal:      "G5_5",
			StepCost:  -1.5, 
			GoalValue: 20.0,
		},
		ActionSpace: mdp.DiscreteActionSpace{
			Action: []mdp.Action{"up", "down", "left", "right"},
		},
		InitialState: "S0_0",
	}
}


type Result struct {
	Success bool
	Steps   int
	Reward  float64
}

func simulate(env *mdp.MDP, opts newrep.Options, episodes int) []Result {
	model := newrep.GridWorldAdapter{Env: env}
	results := make([]Result, 0, episodes)
	for i := 0; i < episodes; i++ {
		s := env.InitialState
		totalReward := 0.0
		steps := 0
		success := false
		for steps < 40 {
			a, _ := newrep.Search(model, s, opts)
			next, r, done := model.Step(s, a)
			totalReward += r
			s = next
			steps++
			if done {
				success = true
				break
			}
		}
		results = append(results, Result{Success: success, Steps: steps, Reward: totalReward})
	}
	return results
}

func avgStats(rs []Result) (succRate float64, avgSteps float64, avgReward float64) {
	totalSucc := 0.0
	totalSteps := 0.0
	totalReward := 0.0
	for _, r := range rs {
		if r.Success {
			totalSucc++
		}
		totalSteps += float64(r.Steps)
		totalReward += r.Reward
	}
	n := float64(len(rs))
	return totalSucc / n, totalSteps / n, totalReward / n
}

func main() {
	rand.Seed(time.Now().UnixNano())
	env := BuildGridWorldEnv()

	baseOpts := newrep.Options{
		C:               math.Sqrt2,
		MaxIterations:   1000,
		MaxRolloutDepth: 60,
		Prior:           newrep.UniformPrior{},
	}

	/*llmOpts := newrep.Options{
		C:               math.Sqrt2,
		MaxIterations:   1000,
		MaxRolloutDepth: 60,
		Prior: newrep.OpenAIChatPrior{
			Model:  "gpt-4o-mini",
			APIKey: os.Getenv("OPEN_API_KEY"), //comment because want to protect API and you can use API here
		},
	}*/

	fmt.Println("Running 10 trials each...")

	baseRes := simulate(env, baseOpts, 10)
	/*llmRes := simulate(env, llmOpts, 10)*/

	bSucc, bSteps, bRew := avgStats(baseRes)
	/*lSucc, lSteps, lRew := avgStats(llmRes)*/

	fmt.Println("\n==== Comparison (Vanilla vs LLM-MCTS) ====")
	fmt.Printf("Baseline MCTS: success=%.2f%% avgSteps=%.1f avgReward=%.2f\n", bSucc*100, bSteps, bRew)
	/*fmt.Printf("LLM-MCTS     : success=%.2f%% avgSteps=%.1f avgReward=%.2f\n", lSucc*100, lSteps, lRew)*/
}
