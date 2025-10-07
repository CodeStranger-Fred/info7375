package newrep

import (
	"github.com/CodeStranger-Fred/assignemnt1/mdp"
)

type Model interface {
	Actions(s mdp.State) []mdp.Action
	Step(s mdp.State, a mdp.Action) (next mdp.State, reward float64, terminal bool)
	IsTerminal(s mdp.State) bool
}

type GridWorldAdapter struct {
	Env *mdp.MDP
}

func (m GridWorldAdapter) Actions(s mdp.State) []mdp.Action {
	if aSpace, ok := m.Env.ActionSpace.(mdp.DiscreteActionSpace); ok {
		return aSpace.Action
	}
	return nil
}

func (m GridWorldAdapter) Step(s mdp.State, a mdp.Action) (mdp.State, float64, bool) {
	s1Pdf := m.Env.TransitionFunction.Transition(s, a).(mdp.DiscretePdf[mdp.State])
	next := s1Pdf.Choose()
	rPdf := m.Env.RewardFunction.Reward(s, a, next).(mdp.DiscretePdf[mdp.Reward])
	r := float64(rPdf.Choose())
	isTerminal := m.IsTerminal(next)
	return next, r, isTerminal
}

func (m GridWorldAdapter) IsTerminal(s mdp.State) bool {
	if len(s) > 0 && (s[0] == 'G' || s[0] == 'T') {
		return true
	}
	return false
}
