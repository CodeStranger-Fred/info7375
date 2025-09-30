package mdp

func MCOnPolicyControl(mdp *MDP, episodes int, epsilon float64) DiscreteStateActionValueEstimator {
	Q := DiscreteStateActionValueEstimator{}
	returns := map[State]map[Action][]float64{}

	for _, s := range mdp.StateSpace.States {
		Q[s] = map[Action]float64{}
		returns[s] = map[Action][]float64{}
		for _, a := range mdp.ActionSpace.Actions(s) {
			Q[s][a] = 0.0
		}
	}

	for ep := 0; ep < episodes; ep++ {
		policy := PolicyEpsilonGreedy{Q: Q, Epsilon: epsilon}
		episode := GenerateEpisode(mdp, policy)

		G := 0.0
		visited := map[[2]string]bool{}
		for i := len(episode) - 1; i >= 0; i-- {
			step := episode[i]
			G = float64(step.Reward) + mdp.RewardDiscount*G
			key := [2]string{string(step.State0), string(step.Action)}
			if !visited[key] {
				returns[step.State0][step.Action] = append(returns[step.State0][step.Action], G)
				sum := 0.0
				for _, v := range returns[step.State0][step.Action] {
					sum += v
				}
				Q[step.State0][step.Action] = sum / float64(len(returns[step.State0][step.Action]))
				visited[key] = true
			}
		}
	}
	return Q
}
