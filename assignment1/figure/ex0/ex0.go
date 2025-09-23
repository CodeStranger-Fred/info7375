package main

import (
	"fmt"
	"math/rand"

	"math"

	. "github.com/CodeStranger-Fred/assignemnt1/mdp"
)


type SelfLoop struct {}

func (b SelfLoop) Transition(state State, action Action) ProbabilityDistribution[State] {
	return DiscretePdf[State]{
		Map: map[State]Probability{
			state: 1,
		},
	}
}



type MultiArmBanditRewardFunction struct {
	ActionReward map[Action]ProbabilityDistribution[Reward]
}

type NormalPdfReward struct {
	Mean float64
	Variance float64
}

func (n NormalPdfReward) Choose() Reward {
	x := rand.NormFloat64()
	y := n.Mean + math.Sqrt(n.Variance)*x
	return Reward(y)
}

func (c MultiArmBanditRewardFunction) Reward(state0 State, action Action, state1 State) ProbabilityDistribution[Reward] {
	return c.ActionReward[action]
}

func RandomBandit(numActions int) MDP {

	actionSpace := DiscreteActionSpace{
		Action: []Action{},
	}

	rewardFunction := MultiArmBanditRewardFunction {
		ActionReward: make(map[Action]ProbabilityDistribution[Reward]),
	}

	for i := 0; i < numActions; i++ {
		action := Action(fmt.Sprintf("a%d", i))
		actionSpace.Action = append(actionSpace.Action, action)
		rewardFunction.ActionReward[action] = NormalPdfReward{
			Mean:rand.NormFloat64(),
			Variance:1,
		}
	}
	//Create an MDP
	mdp := MDP{
		ActionSpace: actionSpace,
		TransitionFunction: SelfLoop{},
		InitialState: State("s0"),
		RewardFunction: rewardFunction,
	}
	return mdp
}


func main() {
	numRuns := 2000
	numSteps := 1000

	mdpGen := func() *MDP {
		mdp := RandomBandit(10)
		return &mdp
	}

	// e-greedy
	rGreedy  := RunPolicyRepeatedly(mdpGen, func() Policy { return Greedy{InitialActionValueEstimate: 0} }, numRuns, numSteps)
	rEps01   := RunPolicyRepeatedly(mdpGen, func() Policy { return GreedyEpsilon{InitialActionValueEstimate: 0, Epsilon: 0.1} }, numRuns, numSteps)
	rEps001  := RunPolicyRepeatedly(mdpGen, func() Policy { return GreedyEpsilon{InitialActionValueEstimate: 0, Epsilon: 0.01} }, numRuns, numSteps)

	// gradient
	rGradA01 := RunPolicyRepeatedly(mdpGen, func() Policy { return &GradientBandit{Alpha: 0.1, UseBaseline: false} }, numRuns, numSteps)
	rGradA04 := RunPolicyRepeatedly(mdpGen, func() Policy { return &GradientBandit{Alpha: 0.4, UseBaseline: false} }, numRuns, numSteps)
	rGradB01 := RunPolicyRepeatedly(mdpGen, func() Policy { return &GradientBandit{Alpha: 0.1, UseBaseline: true} },  numRuns, numSteps)
	rGradB04 := RunPolicyRepeatedly(mdpGen, func() Policy { return &GradientBandit{Alpha: 0.4, UseBaseline: true} },  numRuns, numSteps)

	Plot(rGreedy, rEps01, rEps001, rGradA01, rGradA04, rGradB01, rGradB04)

}