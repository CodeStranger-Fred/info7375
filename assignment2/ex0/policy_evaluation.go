package main

import (
	"fmt"
	"math"

	. "github.com/CodeStranger-Fred/assignemnt1/mdp"
)



func PolicyEvaluation(V DiscreteStateValueEstimator, policy Policy, mdp *MDP) (DiscreteStateValueEstimator, float64) {
	//V := DiscreteStateValueEstimator{}
	states := mdp.StateSpace.States

	var delta float64
	for _, s0 := range states {
		if mdp.IsTerminal(s0) {
			continue
		}
		v0 := V[s0]
		var v1 Reward
		aPdf := policy.Act(nil, mdp, s0)

		for a, ap := range aPdf.(DiscretePdf[Action]) {
			s1Pdf := mdp.TransitionFunction.Transition(s0, a)
			for s1, s1p := range s1Pdf.(DiscretePdf[State]){
				rPdf := mdp.RewardFunction.Reward(s0, a, s1)
				for r, rp := range rPdf.(DiscretePdf[Reward]) {
					p := float64(ap) * float64(s1p) * float64(rp)
                    g := float64(r) + float64(V[s1])*mdp.RewardDiscount
                    v1 += Reward(p * g)
				}
			}
		}
		V[s0] = v1
		delta = math.Max(math.Abs(float64(v0-v1)),delta)
	}
	fmt.Printf("delta: %fn", delta)

	return V, delta

}