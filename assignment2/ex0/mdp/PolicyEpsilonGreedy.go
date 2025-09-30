package mdp

type PolicyEpsilonGreedy struct {
	Q       DiscreteStateActionValueEstimator
	Epsilon float64
}

func (p PolicyEpsilonGreedy) Name() string { 
	return "epsilon-greedy" 
}

func (p PolicyEpsilonGreedy) Act(agent *Agent, mdp *MDP, s State) ProbabilityDistribution[Action] {
	pdf := DiscretePdf[Action]{}
	actions := mdp.ActionSpace.Actions(s)
	if len(actions) == 0 {
		return pdf
	}
	best := p.Q.Argmax(s)
	for _, a := range actions {
		if a == best {
			pdf[a] = Probability(1.0 - p.Epsilon + p.Epsilon/float64(len(actions)))
		} else {
			pdf[a] = Probability(p.Epsilon / float64(len(actions)))
		}
	}
	return pdf
}
