package mdp

func TDOnPolicySARSA(mdp *MDP, episodes int, alpha float64, epsilon float64) DiscreteStateActionValueEstimator {
	Q := DiscreteStateActionValueEstimator{}

	for _, s := range mdp.StateSpace.States {
		Q[s] = map[Action]float64{}
		for _, a := range mdp.ActionSpace.Actions(s) {
			Q[s][a] = 0.0
		}
	}

	for ep := 0; ep < episodes; ep++ {
		policy := PolicyEpsilonGreedy{Q: Q, Epsilon: epsilon}

		state := mdp.InitialState
		aPdf := policy.Act(nil, mdp, state).(DiscretePdf[Action])
		action := aPdf.Choose()

		for !mdp.IsTerminal(state) {
			s1Pdf := mdp.TransitionFunction.Transition(state, action).(DiscretePdf[State])
			nextState := s1Pdf.Choose()
			rPdf := mdp.RewardFunction.Reward(state, action, nextState).(DiscretePdf[Reward])
			reward := rPdf.Choose()

			a1Pdf := policy.Act(nil, mdp, nextState).(DiscretePdf[Action])
			nextAction := a1Pdf.Choose()

			qsa := Q[state][action]
			tdTarget := float64(reward) + mdp.RewardDiscount*Q[nextState][nextAction]
			Q[state][action] = qsa + alpha*(tdTarget-qsa)

			state = nextState
			action = nextAction
		}
	}
	return Q
}
