package mdp

type ActionSpace interface{
	Actions(State) []Action
}
type StateSpace interface{
	IsSingleton() bool
}

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
	StateSpace DiscreteStateSpace
	InitialState State
	RewardDiscount float64
	Terminal map[State]bool
}

func (m MDP) IsTerminal(state State) bool {
	term, ok := m.Terminal[state]
	return term && ok
}

func (d DiscreteStateSpace) IsSingleton() bool {
    return len(d.States) == 1
}