package mdp

import "fmt"

type GreedyEpsilon struct {
	Epsilon float64
	InitialActionValueEstimate Reward
}

func (g GreedyEpsilon) Name() string {
	return fmt.Sprintf("e-greedy-%f",g.Epsilon)
}

func (g GreedyEpsilon) Act(agent *Agent, mdp *MDP, s0 State) ProbabilityDistribution[Action] {
	estimator := SampleAverage(mdp.ActionSpace, agent.History, g.InitialActionValueEstimate)

	maxAction := estimator.Argmax()
	pdf := DiscretePdf[Action] {
		Map: map[Action]Probability{
			maxAction: Probability(1.0 - g.Epsilon),
		},
	}

	das := mdp.ActionSpace.(DiscreteActionSpace)
	numNonMaxActions := len(das.Action) - 1

	for _, action := range das.Action {
		if action != maxAction {
			pdf.Map[action] = Probability(g.Epsilon / float64(numNonMaxActions))
		}
	}
	return pdf
}

