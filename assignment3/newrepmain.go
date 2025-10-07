package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/CodeStranger-Fred/assignemnt1/mdp"
)

type GridWorldTransition struct {
	Width, Height int
	Goal          mdp.State
	Obstacles     map[string]bool
}

func (t GridWorldTransition) Transition(s mdp.State, a mdp.Action) mdp.ProbabilityDistribution[mdp.State] {
	var nx, ny int
	fmt.Sscanf(string(s), "S%d_%d", &nx, &ny)
	switch a {
	case "up":
		ny--
	case "down":
		ny++
	case "left":
		nx--
	case "right":
		nx++
	}
	if nx < 0 || nx >= t.Width || ny < 0 || ny >= t.Height {
		nx, ny = nx, ny
	}
	ns := fmt.Sprintf("S%d_%d", nx, ny)
	if t.Obstacles[ns] {
		ns = string(s)
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
		return mdp.DiscretePdf[mdp.Reward]{Map: map[mdp.Reward]mdp.Probability{mdp.Reward(r.GoalValue): 1.0}}
	}
	return mdp.DiscretePdf[mdp.Reward]{Map: map[mdp.Reward]mdp.Probability{mdp.Reward(r.StepCost): 1.0}}
}

func BuildGridWorldEnv() *mdp.MDP {
	width, height := 5, 5
	obstacles := map[string]bool{
		"S1_1": true, "S2_2": true, "S3_3": true,
	}
	return &mdp.MDP{
		TransitionFunction: GridWorldTransition{
			Width:     width,
			Height:    height,
			Goal:      "G4_4",
			Obstacles: obstacles,
		},
		RewardFunction: GridWorldReward{
			Goal:      "G4_4",
			StepCost:  -1.0,
			GoalValue: 10.0,
		},
		ActionSpace: mdp.DiscreteActionSpace{
			Action: []mdp.Action{"up", "down", "left", "right"},
		},
		InitialState: "S0_0",
	}
}

func newrepmain() {
	rand.Seed(time.Now().UnixNano())
	env := BuildGridWorldEnv()
	model := mcts.GridWorldAdapter{Env: env}

	opts := mcts.Options{
		C:               math.Sqrt2,
		MaxIterations:   3000,
		MaxRolloutDepth: 30,
		Prior:           mcts.UniformPrior{},
	}

	bestAct, info := mcts.Search(model, env.InitialState, opts)
	fmt.Println("Best action:", bestAct)
	fmt.Println(info)
}
