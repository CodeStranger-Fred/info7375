package mdp

type RunStats struct {
	Rewards []Reward
}

type PolicyAverageReward struct {
	Policy         Policy
	AverageRewards []Reward
}

// Single run with a given policy
func RunPolicy(mdp *MDP, policy Policy, numSteps int) RunStats {
	agent := Agent{Policy: policy}
	Loop(mdp, &agent, numSteps)

	stats := RunStats{Rewards: make([]Reward, 0, numSteps)}
	for _, e := range agent.History {
		stats.Rewards = append(stats.Rewards, e.Reward)
	}
	return stats
}

// Multiple runs using a policy factory; each run uses a fresh policy instance
func RunPolicyRepeatedly(
	mdpGen func() *MDP,
	newPolicy func() Policy, // factory: returns a fresh policy instance
	numRuns, numSteps int,
) PolicyAverageReward {

	all := make([]RunStats, 0, numRuns)

	// Create one instance only for labeling in the returned result
	p0 := newPolicy()

	for i := 0; i < numRuns; i++ {
		mdp := mdpGen()
		pol := newPolicy() // new policy per run to avoid cross-run state contamination
		stats := RunPolicy(mdp, pol, numSteps)
		all = append(all, stats)
	}

	avg := make([]Reward, numSteps)
	for t := 0; t < numSteps; t++ {
		sum := 0.0
		for i := 0; i < numRuns; i++ {
			sum += float64(all[i].Rewards[t])
		}
		avg[t] = Reward(sum / float64(numRuns))
	}

	return PolicyAverageReward{
		Policy:         p0,
		AverageRewards: avg,
	}
}