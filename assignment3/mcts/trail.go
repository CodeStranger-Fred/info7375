package mcts

import "github.com/CodeStranger-Fred/assignemnt1/mdp"

type Model interface {
	Actions(s mdp.State) []mdp.Action
	Step(s mdp.State, a mdp.Action) (mdp.State, float64, bool)
	CurrentPlayer(s mdp.State) int
}