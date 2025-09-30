package main

import (
	"fmt"

	. "github.com/CodeStranger-Fred/assignemnt1/mdp"
)

func main() {
	gridworld := StochasticWindyGridWorld{
		Rows:            4,
		Cols:            4,
		BaseWind:        []int{0, 0, 0, 0},
		StochasticWind0: 1.0,
		StochasticWind1: 0.0,
		StochasticWind2: 0.0,
	}
	gridworld.Check()
	mdp := gridworld.MDP()
	mdp.InitialState = gridworld.State(3, 0)
	mdp.RewardDiscount = 0.9

	Q := TDOffPolicyWeightedIS(&mdp, 5000, 0.1, 0.1)
	policy := PolicyGreedy{Estimator: Q}

	gridworld.PrintPolicy(policy)
	fmt.Println("TD(0) off-policy control (Weighted IS) done.")
}