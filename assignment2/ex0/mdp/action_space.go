package mdp



type DiscreteStateSpace struct {
    States []State
}


type DiscreteActionSpace struct {
	Mapping map[State][]Action
}

func (das DiscreteActionSpace) Actions(s State) []Action {
	return das.Mapping[s]
}
