package mdp

type DiscreteActionSpace struct {
    Actions map[State][]Action
}

type DiscreteStateSpace struct {
    States []State
}

