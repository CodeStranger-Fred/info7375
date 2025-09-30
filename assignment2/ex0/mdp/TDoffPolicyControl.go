package mdp

func TDOffPolicyQLearning(mdp *MDP, episodes int, alpha float64, epsilon float64) DiscreteStateActionValueEstimator {
	Q := DiscreteStateActionValueEstimator{}

	for _, s := range mdp.StateSpace.States {
		Q[s] = map[Action]float64{}
		for _, a := range mdp.ActionSpace.Actions(s) {
			Q[s][a] = 0.0
		}
	}

	for ep := 0; ep < episodes; ep++ {
		state := mdp.InitialState
		for !mdp.IsTerminal(state) {
			
			behavior := PolicyEpsilonGreedy{Q: Q, Epsilon: epsilon}
			aPdf := behavior.Act(nil, mdp, state).(DiscretePdf[Action])
			action := aPdf.Choose()

		
			s1Pdf := mdp.TransitionFunction.Transition(state, action).(DiscretePdf[State])
			nextState := s1Pdf.Choose()
			rPdf := mdp.RewardFunction.Reward(state, action, nextState).(DiscretePdf[Reward])
			reward := rPdf.Choose()

			
			bestNext := Q[nextState][Action("")]
			first := true
			for a2, q := range Q[nextState] {
				if first || q > bestNext {
					bestNext = q
					first = false
				}
			}

			qsa := Q[state][action]
			tdTarget := float64(reward) + mdp.RewardDiscount*bestNext
			Q[state][action] = qsa + alpha*(tdTarget-qsa)

			state = nextState
		}
	}
	return Q
}