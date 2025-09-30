package mdp

import "math"

type PolicyGreedy struct {
	Estimator StateActionValueEstimator
}

type StateActionValueEstimator interface {
	Argmax(State) Action
}

type DiscreteStateActionValueEstimator map[State]map[Action]float64

func (g PolicyGreedy) Name() string {return "greedy state-value estimator"}

func (g PolicyGreedy) Act(agent *Agent, mdp *MDP, s0 State) ProbabilityDistribution[Action] {
	return DiscretePdf[Action] {
		g.Estimator.Argmax(s0): 1,
	}
}

func (e DiscreteStateValueEstimator) ToStateActionEstimator(mdp *MDP) DiscreteStateActionValueEstimator {
	stateEstimator := e
	dss := mdp.StateSpace

	estimaor := DiscreteStateActionValueEstimator{}

	for _, s0 := range dss.States {
		estimaor[s0] = make(map[Action]float64)
	}

	for _, s0 := range dss.States {
		for _, a := range mdp.ActionSpace.Actions(s0) {
			txPdf := mdp.TransitionFunction.Transition(s0, a)
			txDpdf := txPdf.(DiscretePdf[State])
			for s1, s1p := range txDpdf {
				s1r := stateEstimator.Estimate(s1)
				estimaor[s0][a] += s1r * float64(s1p)
			}
		}
	}
	return estimaor
}

func (q DiscreteStateActionValueEstimator) Argmax(s State) Action {
	bestA := Action("")
	bestV := math.Inf(-1)
	for a, v := range q[s] {
		if v > bestV {
			bestV = v
			bestA = a
		}
	}
	return bestA
}
