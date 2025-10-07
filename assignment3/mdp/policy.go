package mdp

type Policy interface {
	Name() string

	Act(*Agent, *MDP, State) ProbabilityDistribution[Action]
}


type ActionValueEstimator interface {
	Argmax() Action
	Estimate(Action) Reward
}
