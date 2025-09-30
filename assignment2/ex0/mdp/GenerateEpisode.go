package mdp

func GenerateEpisode(mdp *MDP, policy Policy) []Transition {
	var episode []Transition
	state := mdp.InitialState
	for !mdp.IsTerminal(state) {
		aPdf := policy.Act(nil, mdp, state).(DiscretePdf[Action])
		action := aPdf.Choose()
		s1Pdf := mdp.TransitionFunction.Transition(state, action).(DiscretePdf[State])
		nextState := s1Pdf.Choose()
		rPdf := mdp.RewardFunction.Reward(state, action, nextState).(DiscretePdf[Reward])
		reward := rPdf.Choose()

		episode = append(episode, Transition{
			State0: state,
			Action: action,
			State1: nextState,
			Reward: reward,
		})

		state = nextState
	}
	return episode
}