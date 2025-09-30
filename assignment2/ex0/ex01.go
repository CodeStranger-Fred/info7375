package main


func main() {
	gridworld := StochasticWindyGridWorld{
		Rows: 4,
		Cols: 4,
		BaseWind: []int{1,2,2,1},
		StochasticWind0: 0.1,
		StochasticWind1: 0.8,
		StochasticWind2: 0.1,
	}
	gridworld.Check()

	s := gridworld.State(3,1)
	gridworld.PrintCurrentState(s)

	txpdf := gridworld.Transition(s, "right")
	s = txpdf.Choose()
	gridworld.PrintCurrentState(s)

	txpdf = gridworld.Transition(s, "up")
	s = txpdf.Choose()
	gridworld.PrintCurrentState(s)
}