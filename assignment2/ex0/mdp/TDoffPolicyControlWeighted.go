package mdp

func TDOffPolicyWeightedIS(mdp *MDP, episodes int, alpha float64, epsilon float64) DiscreteStateActionValueEstimator {
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

			target := PolicyGreedy{Estimator: Q}
			greedyAction := target.Estimator.Argmax(state)

			var w float64
			if action == greedyAction {
				// π(a|s) = 1
				// b(a|s) = ε/|A| + (1-ε) if greedy
				actions := mdp.ActionSpace.Actions(state)
				bProb := epsilon/float64(len(actions)) + (1.0-epsilon)
				w = 1.0 / bProb
			} else {
				// π(a|s) = 0
				w = 0.0
			}

			a1 := target.Estimator.Argmax(nextState)
			tdTarget := float64(reward) + mdp.RewardDiscount*Q[nextState][a1]

			qsa := Q[state][action]
			Q[state][action] = qsa + alpha*w*(tdTarget-qsa)

			state = nextState
		}
	}

	return Q
}
