package mdp

type Greedy struct {
	InitialActionValueEstimate Reward
}

func (g Greedy) Name() string {
	return "greedy"
}

func (g Greedy) Act(agent *Agent,mdp *MDP,s0 State) ProbabilityDistribution[Action] {
	estimator := SampleAverage(mdp.ActionSpace, agent.History, g.InitialActionValueEstimate)

	
	return DiscretePdf[Action]{
		Map: map[Action]Probability{
			estimator.Argmax(): 1,
		},
	}
}


