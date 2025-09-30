package mdp

func MCOffPolicyControl(mdp *MDP, episodes int, epsilon float64) DiscreteStateActionValueEstimator {
	Q := DiscreteStateActionValueEstimator{}
	C := map[State]map[Action]float64{}

	for _, s := range mdp.StateSpace.States {
		Q[s] = map[Action]float64{}
		C[s] = map[Action]float64{}
		for _, a := range mdp.ActionSpace.Actions(s) {
			Q[s][a] = 0.0
			C[s][a] = 0.0
		}
	}

	targetPolicy := PolicyGreedy{Estimator: Q}

	for ep := 0; ep < episodes; ep++ {
		behaviorPolicy := PolicyEpsilonGreedy{Q: Q, Epsilon: epsilon}
		episode := GenerateEpisode(mdp, behaviorPolicy)

		G := 0.0
		W := 1.0
		for i := len(episode) - 1; i >= 0; i-- {
			step := episode[i]
			G = float64(step.Reward) + mdp.RewardDiscount*G

			C[step.State0][step.Action] += W
			Q[step.State0][step.Action] += (W / C[step.State0][step.Action]) * (G - Q[step.State0][step.Action])

			if step.Action != targetPolicy.Estimator.Argmax(step.State0) {
				break
			}

			W = W * 1.0 / float64(len(mdp.ActionSpace.Actions(step.State0)))
		}
	}
	return Q
}