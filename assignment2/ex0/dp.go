package main


func dp() {
	gridworld := StochasticWindyGridWorld{
		Rows: 4,
		Cols: 4,
		BaseWind: []int{1,2,2,1},
		StochasticWind0: 0.1,
		StochasticWind1: 0.8,
		StochasticWind2: 0.1,
	}
	gridworld.Check()

	mdp := gridworld.MDP()

	gridworld.PrintValueEstimates(DiscreteStateValueEstimator{})
}