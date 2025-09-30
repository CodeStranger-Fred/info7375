package main

import (
	"fmt"
	"math"
	"strconv"

	. "github.com/CodeStranger-Fred/assignemnt1/mdp"
	"github.com/logrusorgru/aurora"
)

type StochasticWindyGridWorld struct {
	Rows int
	Cols int
	BaseWind []int
	StochasticWind0 Probability
	StochasticWind1 Probability
	StochasticWind2 Probability
}

func (w StochasticWindyGridWorld) MDP() MDP {
	mdp := MDP{}

	dss := DiscreteStateSpace{}
	for s := 0; s < w.Rows*w.Cols; s++ {
		dss.States = append(dss.States, State(strconv.Itoa(s)))
	}
	mdp.StateSpace = dss

	das := DiscreteActionSpace{}
	for _, s := range dss.States {
		das.Actions[s] = []Action{"left", "right", "up", "down"}
	}
	mdp.ActionSpace = das

	mdp.Terminal = map[State]bool{
		w.State(0,0): true,
		w.State(w.Rows-1, w.Cols-1): true,
	}
	mdp.TransitionFunction = w
	mdp.RewardFunction = w

	return mdp
}


func FloatEq(a, b float64) bool {
    const eps = 1e-9
    return math.Abs(a-b) < eps
}

func (w StochasticWindyGridWorld) Check() {
	if !FloatEq(float64(w.StochasticWind0+w.StochasticWind1+w.StochasticWind2), 1) {
		panic("bad prob")
	}
}

func (w StochasticWindyGridWorld) Transition(s0 State, action Action) ProbabilityDistribution[State] {
	s1 := w.Shift(s0,action)
	s1wind0 := s1
	s1wind1 := w.State(w.ClipRow(w.Row(s1) - (w.BaseWind[w.Col(s1)])),w.Col(s1))
	s1wind2 := w.State(w.ClipRow(w.Row(s1) - (w.BaseWind[w.Col(s1)]+1)),w.Col(s1))

	pdf := DiscretePdf[State]{}
	
	Add(&pdf, s1wind0, w.StochasticWind0)
	Add(&pdf, s1wind1, w.StochasticWind1)
	Add(&pdf, s1wind2, w.StochasticWind2)

	return pdf
}

func (w StochasticWindyGridWorld) State(r int, c int) State {
	if r < 0 || c < 0 || r >= w.Rows || r >= w.Cols {
		panic("off board")
	}
	return State(strconv.Itoa(r*w.Cols + c))
}

func (w StochasticWindyGridWorld) Reward(s0 State, a Action, s1 State) ProbabilityDistribution[Reward] {
	return DiscretePdf[Reward]{
		Reward(-1.0): Probability(1.0),
	}
}

func Add[T comparable](pdf *DiscretePdf[T], outcome T, p Probability) {
    if *pdf == nil { 
        *pdf = make(DiscretePdf[T])
    }
    (*pdf)[outcome] += p
}


func (w StochasticWindyGridWorld) Col(s1 State) int {
	_, c1 := w.ToCoordinates(s1)
	return c1
}

func (w StochasticWindyGridWorld) Row(s1 State) int {
	r1, _ := w.ToCoordinates(s1)
	return r1
}

func (w StochasticWindyGridWorld) Shift(s0 State, action Action) State {
	r0, c0 := w.ToCoordinates(s0)

	var r1, c1 int
	switch action {
	case "up":
		r1 = r0 - 1
		c1 = c0
	case "down":
		r1 = r0 + 1
		c1 = c0
	case "right":
		r1 = r0
		c1 = c0 + 1
	case "left":
		r1 = r0
		c1 = c0 - 1
	default:
		panic("unhandled action: " + action)
	}

	r1 = w.ClipRow(r1)
	c1 = w.ClipCol(c1)

	return w.State(r1, c1)
}

var chk = func(err error) {
	if err != nil {panic(err)}
}

func (w StochasticWindyGridWorld) ToCoordinates(s0 State) (int, int) {
	s0n, err := strconv.Atoi(string(s0))
	chk(err)

	if s0n >= w.Rows*w.Cols {
		panic("bad state")
	}

	r := s0n / w.Cols
	c := s0n % w.Cols
	return  r, c
}

func (w StochasticWindyGridWorld) ClipRow(r1 int) int {
	if r1 < 0 {
		r1 = 0
	}
	if r1 > w.Rows - 1{
		r1 = w.Rows - 1
	}

	return r1
}

func (w StochasticWindyGridWorld) ClipCol(c1 int) int {
	if c1 < 0 {
		c1 = 0
	}
	if c1 > w.Cols - 1 {
		c1 = w.Cols - 1
	}

	return c1
}

func (w StochasticWindyGridWorld) PrintCurrentState(currentState State) {
	for r := 0; r < w.Rows; r++ {
		for c := 0; c < w.Cols; c++ {
			st := w.State(r,c)
			if st == currentState {
				fmt.Print(aurora.Green(fmt.Sprintf("%5s ", st)))
			} else {
				fmt.Print(aurora.Blue(fmt.Sprintf("%5s ", st)))
			}
			fmt.Print(aurora.White(fmt.Sprintf("|")))
		}
		fmt.Printf("\n")
	}
}

func (w StochasticWindyGridWorld) PrintValueEstimates(estimator agent.StateValueEstimator) {
	for r := 0; r < w.Rows; r++ {
		for c := 0; c < w.Cols; c++ {
			st := w.State(r,c)
			v := estimator.Estimate(st)
			fmt.Print(aurora.Blue(format2x2(float64(v))))
			fmt.Print((aurora.White((fmt.Sprintf("|")))))
		}
		fmt.Printf(("\n"))
	}
}

func format2x2(x float64) string {
	if x < 0 {
		return " -" + fmt.Sprintf("%05.2f", -x)
	}
	return fmt.Sprintf(" %05.2f", x)
}