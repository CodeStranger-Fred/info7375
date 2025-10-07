package mdp

type Transition struct {
	State0 State
	Action Action
	State1 State
	Reward Reward
}

type Agent struct {
	History []Transition 
	Policy Policy
}

func (a *Agent) Step(state0 State, action Action, state1 State, reward Reward) {
	a.History = append(a.History, Transition{
		State0: state0,
		Action: action,
		State1: state1,
		Reward: reward,
	})
}


