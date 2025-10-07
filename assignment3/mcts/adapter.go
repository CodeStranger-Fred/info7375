package mcts

import "github.com/CodeStranger-Fred/assignemnt1/mdp"

type ActionProvider func(s mdp.State) []mdp.Action

type MDPAdapter struct {
	Env      *mdp.MDP
	ActProv  ActionProvider
}

func (m *MDPAdapter) Actions(s mdp.State) []mdp.Action {
	if m.ActProv != nil {
		return m.ActProv(s)
	}
	if das, ok := m.Env.ActionSpace.(mdp.DiscreteActionSpace); ok {
		out := make([]mdp.Action, len(das.Action))
		copy(out, das.Action)
		return out
	}
	return nil
}

func (m *MDPAdapter) Step(s mdp.State, a mdp.Action) (mdp.State, float64, bool) {
	s1 := m.Env.TransitionFunction.Transition(s, a).Choose()
	r := m.Env.RewardFunction.Reward(s, a, s1).Choose()
	term := len(m.Actions(s1)) == 0
	return s1, float64(r), term
}

func (m *MDPAdapter) CurrentPlayer(s mdp.State) int {
	return 1
}
