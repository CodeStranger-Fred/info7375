package main

import (
	"fmt"

	. "github.com/CodeStranger-Fred/assignemnt1/mdp"
)
type PolicyRandom struct {

}

func (p PolicyRandom) Name() string {
	return "random"
}

func (p PolicyRandom) Act(agent *Agent, mdp *MDP, s0 State) ProbabilityDistribution[Action] {
	pdf := DiscretePdf[Action]{}
	actions := mdp.ActionSpace.Actions(s0)
	for _, a := range actions {
		pdf[a] = Probability(1.0 / float64(len(actions)))
	}
	return pdf
}

func main() {
	gridworld := StochasticWindyGridWorld{
		Rows: 4,
		Cols: 4,
		BaseWind: []int{0,0,0,0},
		StochasticWind0: 1.0,
		StochasticWind1: 0.0,
		StochasticWind2: 0.0,
	}
	gridworld.Check()

	mdp := gridworld.MDP()

	V := DiscreteStateValueEstimator{}
	gridworld.PrintValueEstimates(V)

	var policy Policy

	policy = PolicyRandom{}
	gridworld.PrintPolicy(policy)

	k := 0

	for {
		var delta float64
	V, delta = PolicyEvaluation(V, policy, &mdp)
	fmt.Println("\n")
	gridworld.PrintValueEstimates(V)

	stateValueEstimator := V
	stateActionEstimator := stateValueEstimator.ToStateActionEstimator(&mdp)


	policy = PolicyGreedy{
		Estimator: stateActionEstimator,
	}

	gridworld.PrintPolicy(policy)

	if delta < .1 {
		break
	}
	k++

	}
}