package mdp

type ActionSpace interface{}
type StateSpace interface{}

type State string

type Action string

type Reward float64

type TransitionFunction interface {
	Transition(State, Action) ProbabilityDistribution[State]
}

type RewardFunction interface {
	Reward(State, Action, State) ProbabilityDistribution[Reward]
}

type MDP struct {
	TransitionFunction TransitionFunction
	RewardFunction RewardFunction
	ActionSpace ActionSpace
	InitialState State
}