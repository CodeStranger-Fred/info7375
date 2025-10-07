package mdp

type RewardStats struct {
	Initial Reward
	Sum Reward
	Avg Reward
	Count int
}

func (stats *RewardStats) Add(r Reward) {
	stats.Sum += r
	stats.Count++
	stats.Avg = Reward(float64(stats.Sum) / float64(stats.Count))
}

type DiscreteActionValueEstimator struct {
	Map map[Action]*RewardStats
}

func (e DiscreteActionValueEstimator) Estimate(action Action) Reward {
	if e.Map[action].Count == 0 {
		return e.Map[action].Initial
	} else {
		return e.Map[action].Avg
	}
}

func (e DiscreteActionValueEstimator) Argmax() Action {
	if e.Map == nil || len(e.Map) == 0 {panic("unfilled map")}

	var argmaxAction Action
	var argmaxReward Reward
	for a := range e.Map {
		r := e.Estimate(a)
		if argmaxAction == "" || argmaxReward < r {
			argmaxAction = a
			argmaxReward = r
		}
	}
	return argmaxAction
}

func SampleAverage(space ActionSpace, history []Transition, initial Reward) ActionValueEstimator {

	discreteActionSpace := space.(DiscreteActionSpace)
	e := DiscreteActionValueEstimator{
		Map: make(map[Action]*RewardStats),
	}
	for _, action := range discreteActionSpace.Action {
		e.Map[action] = &RewardStats{
			Initial: initial,
		}
	}

	for _, transition := range history {
		e.Map[transition.Action].Add(transition.Reward)
	}

	return e
}

